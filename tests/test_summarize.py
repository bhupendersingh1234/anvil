import json
from types import SimpleNamespace

from anvil.data.summarize import BATCH_SIZE, BatchRunner, Preprocessor, batch_line, summarize_items


class FakeGroq:
    def __init__(self, items):
        self.items = items
        self.files = SimpleNamespace(create=self.create_file, content=self.content)
        self.batches = SimpleNamespace(create=self.create_batch, retrieve=self.retrieve)
        self.uploaded = {}

    def create_file(self, file, purpose):
        file_id = f"file_{len(self.uploaded)}"
        self.uploaded[file_id] = file.read().decode().splitlines()
        return SimpleNamespace(id=file_id)

    def create_batch(self, completion_window, endpoint, input_file_id):
        return SimpleNamespace(id=f"batch_{input_file_id}")

    def retrieve(self, batch_id):
        return SimpleNamespace(status="completed", output_file_id=batch_id.removeprefix("batch_"))

    def content(self, file_id):
        lines = [json.loads(line) for line in self.uploaded[file_id]]
        output = "\n".join(
            json.dumps({"custom_id": row["custom_id"], "response": {"body": {"choices": [{"message": {"content": f"summary {row['custom_id']}"}}]}}})
            for row in lines
        )
        return SimpleNamespace(write_to_file=lambda path: open(path, "w").write(output))


def test_batch_line_format(items) -> None:
    line = json.loads(batch_line(7, items[0]))
    assert line["custom_id"] == "7" and line["body"]["messages"][1]["content"] == items[0].full


def test_batch_runner_submit_fetch_and_resume(tmp_path, items) -> None:
    items = (items * 5)[: BATCH_SIZE + 5]
    items = [item.model_copy(update={"summary": None}) for item in items]
    runner = BatchRunner(items, tmp_path, client=FakeGroq(items))
    assert runner.submit() == 2
    assert runner.submit() == 0
    assert runner.fetch() == 2
    assert items[BATCH_SIZE + 4].summary == f"summary {BATCH_SIZE + 4}"
    fresh = [item.model_copy(update={"summary": None}) for item in items]
    resumed = BatchRunner(fresh, tmp_path, client=FakeGroq(fresh))
    assert all(batch.done for batch in resumed.batches)
    resumed.apply_all()
    assert fresh[3].summary == "summary 3"


def test_summarize_items_skips_existing(items) -> None:
    preprocessor = Preprocessor("ollama/llama3.2")
    preprocessor.preprocess = lambda text: "rewritten"
    items = [item.model_copy(update={"summary": None}) for item in items[:5]]
    items[0].summary = "keep"
    summarize_items(items, preprocessor, workers=2)
    assert [item.summary for item in items] == ["keep"] + ["rewritten"] * 4


def test_preprocessor_defaults() -> None:
    assert Preprocessor("ollama/llama3.2").base_url == "http://localhost:11434"
    assert Preprocessor("groq/openai/gpt-oss-20b").reasoning_effort == "low"
