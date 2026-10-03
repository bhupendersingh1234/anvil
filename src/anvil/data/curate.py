import random

import numpy as np

from anvil.data.items import Item

SAMPLE_SIZE = 820_000
CATEGORY_WEIGHTS = {"Tools_and_Home_Improvement": 0.5, "Automotive": 0.05}


def unique_by(items: list[Item], field: str) -> list[Item]:
    seen = set()
    return [item for item in items if not (getattr(item, field) in seen or seen.add(getattr(item, field)))]


def deduplicate(items: list[Item], seed: int = 42) -> list[Item]:
    items = items[:]
    random.Random(seed).shuffle(items)
    return unique_by(unique_by(items, "title"), "full")


def sample_weights(items: list[Item]) -> np.ndarray:
    prices = np.array([item.price for item in items], dtype=float)
    categories = np.array([item.category for item in items])
    weights = ((prices - prices.min()) / (prices.max() - prices.min() + 1e-9)) ** 2
    for category, factor in CATEGORY_WEIGHTS.items():
        weights[categories == category] *= factor
    return weights / weights.sum()


def weighted_sample(items: list[Item], size: int = SAMPLE_SIZE, seed: int = 42) -> list[Item]:
    weights = sample_weights(items)
    size = min(size, int(np.count_nonzero(weights)))
    indices = np.random.RandomState(seed).choice(len(items), size=size, replace=False, p=weights)
    sample = [items[i] for i in indices]
    random.Random(seed).shuffle(sample)
    return sample


def split(items: list[Item], sizes: tuple[int, int, int]) -> tuple[list[Item], list[Item], list[Item]]:
    if len(items) < sum(sizes):
        scale = len(items) / sum(sizes)
        val_size, test_size = max(1, int(sizes[1] * scale)), max(1, int(sizes[2] * scale))
        sizes = (len(items) - val_size - test_size, val_size, test_size)
    train_size, val_size, test_size = sizes
    val_end = train_size + val_size
    return items[:train_size], items[train_size:val_end], items[val_end:val_end + test_size]
