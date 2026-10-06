import hashlib

import numpy as np
import pytest

from anvil import rag
from anvil.data.items import Item


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[b / 255 for b in hashlib.sha256(text.encode()).digest()[:16]] for text in texts])


@pytest.fixture
def collection(tmp_path):
    return rag.get_collection(tmp_path / "store")


def test_build_index_is_resumable(collection, items) -> None:
    encoder = FakeEncoder()
    assert rag.build_index(collection, encoder, items[:150], batch_size=64) == 150
    assert rag.build_index(collection, encoder, items, batch_size=64) == len(items)


def test_find_similars_returns_exact_match_first(collection, items) -> None:
    encoder = FakeEncoder()
    rag.build_index(collection, encoder, items)
    documents, prices = rag.find_similars(collection, encoder, items[42].summary)
    assert documents[0] == items[42].summary and prices[0] == items[42].price and len(prices) == 5


def test_messages_include_context() -> None:
    content = rag.messages_for("Mystery widget", ["Similar widget"], [12.5])[0]["content"]
    assert content.startswith("Estimate the price") and "Similar widget\nPrice is $12.50" in content


def test_plot_data(collection) -> None:
    categories = ["Appliances", "Electronics", "Toys_and_Games"]
    items = [Item(title=f"t{i}", category=categories[i % 3], price=10, summary=f"item {i}") for i in range(40)]
    rag.build_index(collection, FakeEncoder(), items)
    documents, vectors, colors = rag.plot_data(collection, max_datapoints=30)
    assert vectors.shape == (30, 3) and len(colors) == len(documents) == 30
