import json
import random

import pytest

from anvil.data.items import Item

WORDS = "steel wireless compact durable premium portable digital battery kitchen audio laptop camera guitar pedal cable charger".split()


def make_item(i: int, price: float | None = None, category: str = "Electronics") -> Item:
    rng = random.Random(i)
    words = " ".join(rng.choices(WORDS, k=12))
    price = price if price is not None else round(rng.uniform(5, 900), 2)
    summary = f"Title: Product {i} {words}\nCategory: {category}\nBrand: Acme\nDescription: {words}\nDetails: size {i % 7}"
    return Item(title=f"Product {i}", category=category, price=price, summary=summary, full=summary * 3, weight=rng.choice([0, 1.5, 3.0]))


def raw_datapoint(i: int, price="19.99", details: dict | str | None = None) -> dict:
    return {
        "title": f"Widget {i} deluxe edition",
        "price": price,
        "description": ["A very good widget. " * 20],
        "features": ["Feature one is great", "Feature two is better " * 10],
        "details": details if details is not None else json.dumps({"Item Weight": "2 pounds", "Part Number": "ABC1234567", "Brand": "Acme"}),
    }


@pytest.fixture
def items() -> list[Item]:
    return [make_item(i) for i in range(300)]
