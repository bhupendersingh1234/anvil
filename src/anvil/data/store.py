import json
from pathlib import Path

from anvil import config
from anvil.data.items import Item

SPLITS = ("train", "validation", "test")
Splits = tuple[list[Item], list[Item], list[Item]]


def local_dir(kind: str, lite: bool) -> Path:
    return config.DATA_DIR / f"{kind}_{'lite' if lite else 'full'}"


def save_splits(kind: str, lite: bool, splits: tuple[list, list, list]) -> Path:
    folder = local_dir(kind, lite)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(SPLITS, splits):
        lines = (json.dumps(row.to_datapoint() if kind == "items_prompts" else row.model_dump()) for row in rows)
        (folder / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_splits(kind: str, lite: bool) -> Splits:
    folder = local_dir(kind, lite)
    if (folder / "train.jsonl").exists():
        return tuple([Item.model_validate(row) for row in read_jsonl(folder / f"{name}.jsonl")] for name in SPLITS)
    return Item.from_hub(config.dataset_name(kind, lite))


def lite_subset(splits: Splits) -> Splits:
    return tuple(rows[:size] for rows, size in zip(splits, config.SPLITS[True]))


def push_splits(kind: str, lite: bool, splits: Splits) -> str:
    target = config.push_target(kind, lite)
    (Item.push_prompts_to_hub if kind == "items_prompts" else Item.push_to_hub)(target, *splits)
    return target
