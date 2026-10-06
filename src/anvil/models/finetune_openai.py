import json
from pathlib import Path

from openai import OpenAI

from anvil.data.items import Item
from anvil.models.baselines import Predictor
from anvil.models.frontier import messages_for

BASE_MODEL = "gpt-4.1-nano-2025-04-14"


def training_messages(item: Item) -> list[dict]:
    return messages_for(item.summary) + [{"role": "assistant", "content": f"${item.price:.2f}"}]


def make_jsonl(items: list[Item]) -> str:
    return "\n".join(json.dumps({"messages": training_messages(item)}) for item in items)


def write_jsonl(items: list[Item], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_jsonl(items), encoding="utf-8")
    return path


def upload(client: OpenAI, path: Path) -> str:
    with path.open("rb") as f:
        return client.files.create(file=f, purpose="fine-tune").id


def start_job(client: OpenAI, train: list[Item], val: list[Item], folder: Path, model: str = BASE_MODEL, epochs: int = 1) -> str:
    train_id = upload(client, write_jsonl(train, folder / "fine_tune_train.jsonl"))
    val_id = upload(client, write_jsonl(val, folder / "fine_tune_validation.jsonl"))
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=model,
        seed=42,
        hyperparameters={"n_epochs": epochs, "batch_size": 1},
        suffix="anvil",
    )
    return job.id


def fine_tuned_pricer(client: OpenAI, model: str) -> Predictor:
    def predict(item: Item) -> str:
        return client.chat.completions.create(model=model, messages=messages_for(item.summary), max_tokens=7).choices[0].message.content

    return predict
