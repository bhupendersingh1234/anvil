import json
import socket
import urllib.error

import pytest

from anvil.data import curate, loaders, store
from anvil.data.loaders import load_category
from anvil.data.prompts import build_prompts
from conftest import make_item, raw_datapoint
from test_items import FakeTokenizer


def test_load_category_parses_jsonl_in_parallel(tmp_path) -> None:
    path = tmp_path / "meta.jsonl"
    rows = [raw_datapoint(i, price=10 + i % 900, details={"Brand": "Acme"}) for i in range(2500)] + [raw_datapoint(9999, price=None)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    items = load_category("Appliances", workers=2, path=path)
    assert len(items) == 2500
    assert {item.category for item in items} == {"Appliances"}


class FakeResponse:
    status = 200

    def __init__(self, chunks: list[bytes], length: str):
        self.chunks = iter(chunks)
        self.length = length

    def getheader(self, name: str) -> str | None:
        return self.length if name == "Content-Length" else None

    def read(self, n: int) -> bytes:
        return next(self.chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


def test_fetch_to_file_raises_on_truncated_response(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.urllib.request, "urlopen", lambda request, timeout: FakeResponse([b"x" * 400, b""], "1000"))
    path = tmp_path / "out.jsonl"
    with pytest.raises(OSError, match="Incomplete download"):
        loaders._fetch_to_file("http://example.test/file", path, resume_from=0)
    assert path.read_bytes() == b"x" * 400


def test_download_uses_manually_placed_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    manual = loaders.manual_download_path("Electronics")
    manual.parent.mkdir(parents=True)
    manual.write_text("already have it")
    monkeypatch.setattr(loaders, "_fetch_to_file", lambda *a, **k: pytest.fail("should not fetch"))
    assert loaders.download("Electronics") == manual


def test_download_resumes_from_previous_progress(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    resumes = []

    def flaky(url, path, resume_from):
        resumes.append(resume_from)
        path.write_bytes(b"x" * resume_from + b"y" * 100)
        if len(resumes) < 3:
            raise RuntimeError("connection reset")

    monkeypatch.setattr(loaders, "_fetch_to_file", flaky)
    sleeps = []
    monkeypatch.setattr(loaders.time, "sleep", sleeps.append)
    assert loaders.download("Electronics", retries=5, backoff=1).read_bytes() == b"x" * 200 + b"y" * 100
    assert resumes == [0, 100, 200] and sleeps == [1, 2]


def test_download_gives_up_after_retries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loaders, "_fetch_to_file", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("still broken")))
    monkeypatch.setattr(loaders.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Failed to download"):
        loaders.download("Electronics", retries=3, backoff=0)


def test_download_short_backoff_for_dns_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    calls = []

    def flaky(url, path, resume_from):
        calls.append(1)
        if len(calls) < 3:
            raise socket.gaierror("[Errno 11001] getaddrinfo failed")
        path.write_bytes(b"done")

    monkeypatch.setattr(loaders, "_fetch_to_file", flaky)
    sleeps = []
    monkeypatch.setattr(loaders.time, "sleep", sleeps.append)
    loaders.download("Electronics", retries=5, backoff=50)
    assert sleeps == [loaders.DNS_RETRY_BACKOFF_SECONDS] * 2


def test_download_fails_fast_on_permanent_http_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)

    def not_found(url, path, resume_from):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(loaders, "_fetch_to_file", not_found)
    monkeypatch.setattr(loaders.time, "sleep", lambda seconds: pytest.fail("a 404 should not be retried"))
    with pytest.raises(RuntimeError, match="HTTP 404"):
        loaders.download("Electronics")


def test_download_retries_forever_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    calls = []

    def flaky(url, path, resume_from):
        calls.append(1)
        if len(calls) < 40:
            raise RuntimeError("connection reset")
        path.write_bytes(b"done")

    monkeypatch.setattr(loaders, "_fetch_to_file", flaky)
    monkeypatch.setattr(loaders.time, "sleep", lambda seconds: None)
    assert loaders.download("Electronics").read_bytes() == b"done" and len(calls) == 40


def test_load_cached_category_caches_and_force_reloads(monkeypatch, tmp_path, items) -> None:
    monkeypatch.setattr(loaders.config, "DATA_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(loaders, "load_category", lambda category, workers: calls.append(category) or items[:5])
    assert loaders.load_cached_category("Toys_and_Games") == items[:5]
    assert loaders.load_cached_category("Toys_and_Games") == items[:5]
    loaders.load_cached_category("Toys_and_Games", force=True)
    assert calls == ["Toys_and_Games", "Toys_and_Games"]


def test_deduplicate_by_title_then_full() -> None:
    a, b, c = make_item(1), make_item(2), make_item(3)
    b.full = a.full
    c.title = a.title
    assert curate.deduplicate([a, b, c]) in ([a], [b], [c])


def test_weighted_sample_prefers_expensive_and_dampens_automotive(items) -> None:
    cars = [make_item(1000 + i, price=900, category="Automotive") for i in range(100)]
    sample = curate.weighted_sample(items + cars, size=100)
    assert len(sample) == 100
    assert sum(item.price for item in sample) / 100 > sum(item.price for item in items) / len(items)
    assert sum(item.category == "Automotive" for item in sample) < 50


def test_split_scales_down_for_small_samples(items) -> None:
    train, val, test = curate.split(items, (800_000, 10_000, 10_000))
    assert len(train) + len(val) + len(test) == len(items)
    assert val and test
    assert curate.split(items, (100, 10, 10)) == (items[:100], items[100:110], items[110:120])


def test_store_roundtrip(tmp_path, monkeypatch, items) -> None:
    monkeypatch.setattr(store.config, "DATA_DIR", tmp_path)
    splits = (items[:200], items[200:250], items[250:])
    store.save_splits("items", True, splits)
    assert store.load_splits("items", True) == splits
    assert [len(rows) for rows in store.lite_subset(splits)] == [200, 50, 50]


def test_build_prompts_rounds_train_only(items) -> None:
    train, val, test = items[:10], items[10:20], items[20:30]
    build_prompts(train, val, test, FakeTokenizer(), cutoff=5)
    assert train[0].completion.endswith(".00") and val[0].completion.endswith(".00")
    assert test[0].completion == str(test[0].price)
    assert train[0].prompt.endswith("Price is $")