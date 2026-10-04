import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import litellm
from litellm import completion
from litellm.exceptions import APIConnectionError, InternalServerError, RateLimitError, ServiceUnavailableError, Timeout
from tqdm.auto import tqdm

from anvil import config
from anvil.data.items import Item

litellm.suppress_debug_info = True

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

BATCH_SIZE = 1_000
FALLBACK_MODEL = os.getenv("ANVIL_FALLBACK_MODEL", "groq/openai/gpt-oss-20b")
RETRYABLE = (RateLimitError, Timeout, APIConnectionError, ServiceUnavailableError, InternalServerError)
QUOTA_ERRORS = ("insufficient_quota", "exceeded your current quota", "billing")
MAX_RETRIES = 200
MAX_WAIT = 60.0
WAIT_PATTERN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)
STOP = threading.Event()


def messages_for(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def retry_wait(error: Exception, attempt: int) -> float:
    match = WAIT_PATTERN.search(str(error))
    base = int(match.group(1) or 0) * 60 + float(match.group(2)) + 0.5 if match else min(2.0 * 2 ** (attempt - 1), MAX_WAIT)
    return base + random.uniform(0, 1)


def out_of_credit(error: Exception) -> bool:
    return any(phrase in str(error).lower() for phrase in QUOTA_ERRORS)


class Preprocessor:
    def __init__(
        self,
        model_name: str = config.PREPROCESSOR_MODEL,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
        fallback: str | None = FALLBACK_MODEL,
    ):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort or ("low" if "gpt-oss" in model_name else None)
        self.base_url = base_url or ("http://localhost:11434" if "ollama" in model_name else None)
        self.fallback = fallback
        self.lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def switch_to_fallback(self, failed: str) -> None:
        with self.lock:
            if self.model_name == failed and self.fallback and self.fallback != failed:
                print(f"\n{failed} is out of credit - switching to {self.fallback}")
                self.model_name, self.fallback = self.fallback, None
                self.reasoning_effort = "low" if "gpt-oss" in self.model_name else None
                self.base_url = None

    def call(self, text: str):
        for attempt in range(1, MAX_RETRIES + 1):
            if STOP.is_set():
                raise RuntimeError("stopped")
            model = self.model_name
            try:
                return completion(messages=messages_for(text), model=model, reasoning_effort=self.reasoning_effort, api_base=self.base_url, timeout=120)
            except RETRYABLE as error:
                if out_of_credit(error):
                    self.switch_to_fallback(model)
                    if self.model_name != model:
                        continue
                    raise
                if attempt == MAX_RETRIES:
                    raise
                STOP.wait(retry_wait(error, attempt))
        raise RuntimeError("unreachable")

    def preprocess(self, text: str) -> str:
        response = self.call(text)
        with self.lock:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.total_cost += response._hidden_params.get("response_cost") or 0.0
        return response.choices[0].message.content.strip()


def key(item: Item) -> str:
    return hashlib.sha1(item.full.encode()).hexdigest()[:16]


def load_checkpoint(path: Path) -> dict[str, str]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["key"]] = row["summary"]
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def summarize_items(items: list[Item], preprocessor: Preprocessor, workers: int = 4, checkpoint: Path | None = None) -> None:
    checkpoint = checkpoint or config.DATA_DIR / "summaries.jsonl"
    done = load_checkpoint(checkpoint)
    for item in items:
        item.summary = item.summary or done.get(key(item))
    pending = [item for item in items if not item.summary]
    if not pending:
        return
    print(f"{len(items) - len(pending):,} already summarized, {len(pending):,} to go with {preprocessor.model_name}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    STOP.clear()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        remaining = {pool.submit(lambda item=item: (item, preprocessor.preprocess(item.full))) for item in pending}
        with checkpoint.open("a", encoding="utf-8") as out, tqdm(total=len(pending)) as bar:
            while remaining:
                finished, remaining = wait(remaining, timeout=1, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda f: f.exception() is not None):
                    item, summary = future.result()
                    item.summary = summary
                    out.write(json.dumps({"key": key(item), "summary": summary}) + "\n")
                    out.flush()
                    bar.update()
                    bar.set_postfix(cost=f"${preprocessor.total_cost:.2f}", model=preprocessor.model_name.split("/")[-1])
    finally:
        STOP.set()
        pool.shutdown(wait=True, cancel_futures=True)
        STOP.clear()


def batch_line(index: int, item: Item, model: str = config.BATCH_MODEL) -> str:
    body = {"model": model, "messages": messages_for(item.full), "reasoning_effort": "low"}
    return json.dumps({"custom_id": str(index), "method": "POST", "url": "/v1/chat/completions", "body": body})


@dataclass
class Batch:
    start: int
    end: int
    file_id: str | None = None
    batch_id: str | None = None
    output_file_id: str | None = None
    done: bool = False

    @property
    def filename(self) -> str:
        return f"{self.start}_{self.end}.jsonl"


class BatchRunner:
    def __init__(self, items: list[Item], folder: Path, client=None):
        self.items = items
        self.inputs = folder / "input"
        self.outputs = folder / "output"
        self.state = folder / "batches.json"
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.batches = [Batch(start, min(start + BATCH_SIZE, len(items))) for start in range(0, len(items), BATCH_SIZE)]
        if self.state.exists():
            self.batches = [Batch(**data) for data in json.loads(self.state.read_text())]

    def groq(self):
        if not self.client:
            from groq import Groq

            self.client = Groq()
        return self.client

    def save(self) -> None:
        self.state.write_text(json.dumps([asdict(batch) for batch in self.batches], indent=2))

    def write_input(self, batch: Batch) -> Path:
        path = self.inputs / batch.filename
        path.write_text("\n".join(batch_line(i, self.items[i]) for i in range(batch.start, batch.end)) + "\n", encoding="utf-8")
        return path

    def submit(self) -> int:
        pending = [batch for batch in self.batches if not batch.batch_id]
        for batch in tqdm(pending, desc="Submitting"):
            with self.write_input(batch).open("rb") as f:
                batch.file_id = self.groq().files.create(file=f, purpose="batch").id
            batch.batch_id = self.groq().batches.create(completion_window="24h", endpoint="/v1/chat/completions", input_file_id=batch.file_id).id
            self.save()
        return len(pending)

    def apply_output(self, batch: Batch) -> None:
        for line in (self.outputs / batch.filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            self.items[int(row["custom_id"])].summary = row["response"]["body"]["choices"][0]["message"]["content"]
        batch.done = True

    def fetch(self) -> int:
        for batch in tqdm([b for b in self.batches if b.batch_id and not b.done], desc="Fetching"):
            response = self.groq().batches.retrieve(batch.batch_id)
            if response.status == "completed":
                batch.output_file_id = response.output_file_id
                self.groq().files.content(batch.output_file_id).write_to_file(self.outputs / batch.filename)
                self.apply_output(batch)
        self.save()
        return sum(batch.done for batch in self.batches)

    def apply_all(self) -> None:
        for batch in self.batches:
            if (self.outputs / batch.filename).exists():
                self.apply_output(batch)