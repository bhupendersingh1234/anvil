import re
from typing import Optional, Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"
PRICE_PATTERN = re.compile(r"[-+]?\d*\.\d+|\d+")


def extract_price(value) -> float:
    if not isinstance(value, str):
        return float(value)
    match = PRICE_PATTERN.search(value.replace("$", "").replace(",", ""))
    return float(match.group()) if match else 0.0


def inference_prompt(text: str) -> str:
    return f"{QUESTION}\n\n{text}\n\n{PREFIX}"


class Item(BaseModel):
    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str) -> None:
        self.prompt = f"{inference_prompt(text)}{round(self.price)}.00"

    def test_prompt(self) -> str:
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    def count_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_prompts(self, tokenizer, max_tokens: int, do_round: bool) -> None:
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        summary = tokenizer.decode(tokens[:max_tokens]).rstrip() if len(tokens) > max_tokens else self.summary
        self.prompt = inference_prompt(summary)
        self.completion = f"{round(self.price)}.00" if do_round else str(self.price)

    def count_prompt_tokens(self, tokenizer) -> int:
        return len(tokenizer.encode(self.prompt + self.completion, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        return {"prompt": self.prompt, "completion": self.completion}

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]) -> None:
        DatasetDict({
            "train": Dataset.from_list([item.model_dump() for item in train]),
            "validation": Dataset.from_list([item.model_dump() for item in val]),
            "test": Dataset.from_list([item.model_dump() for item in test]),
        }).push_to_hub(dataset_name)

    @staticmethod
    def push_prompts_to_hub(dataset_name: str, train: list[Self], val: list[Self], test: list[Self]) -> None:
        DatasetDict({
            "train": Dataset.from_list([item.to_datapoint() for item in train]),
            "val": Dataset.from_list([item.to_datapoint() for item in val]),
            "test": Dataset.from_list([item.to_datapoint() for item in test]),
        }).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return tuple([cls.model_validate(row) for row in ds[split]] for split in ("train", "validation", "test"))
