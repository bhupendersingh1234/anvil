from tqdm.auto import tqdm

from anvil.data.items import Item

CUTOFF = 110


def build_prompts(train: list[Item], val: list[Item], test: list[Item], tokenizer, cutoff: int = CUTOFF) -> float:
    for item in tqdm(train + val, desc="Train/val prompts"):
        item.make_prompts(tokenizer, cutoff, True)
    for item in tqdm(test, desc="Test prompts"):
        item.make_prompts(tokenizer, cutoff, False)
    items = train + val + test
    return sum(item.count_tokens(tokenizer) > cutoff for item in items) / len(items)
