import gzip
import json
import os
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import islice
from pathlib import Path

from tqdm.auto import tqdm

from anvil import config
from anvil.data.items import Item
from anvil.data.parser import parse

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/raw/meta_categories/meta_{{category}}.jsonl"
CHUNK_SIZE = 1000
WORKERS = max((os.cpu_count() or 2) - 1, 1)
RETRY_BACKOFF_SECONDS = 10.0
MAX_BACKOFF_SECONDS = 120.0
DNS_RETRY_BACKOFF_SECONDS = 5.0
DOWNLOAD_CHUNK_BYTES = 512 * 1024
SOCKET_TIMEOUT_SECONDS = 30.0
DNS_ERROR_MARKERS = ("getaddrinfo failed", "Name or service not known", "nodename nor servname", "Errno 11001")
FATAL_HTTP_STATUSES = frozenset(range(400, 500)) - {408, 429}


def manual_download_path(category: str) -> Path:
    return config.DATA_DIR / "raw_cache" / "downloads" / f"meta_{category}.jsonl"


def _fetch_to_file(url: str, path: Path, resume_from: int) -> None:
    headers = {"User-Agent": "anvil-downloader/1.0"} | ({"Range": f"bytes={resume_from}-"} if resume_from else {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=SOCKET_TIMEOUT_SECONDS) as response:
        resumed = bool(resume_from) and response.status == 206
        start = resume_from if resumed else 0
        remaining = response.getheader("Content-Length")
        total = start + int(remaining) if remaining is not None else None
        with open(path, "ab" if resumed else "wb") as f, tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=path.stem) as bar:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                f.write(chunk)
                bar.update(len(chunk))
    actual = path.stat().st_size
    if total is not None and actual != total:
        raise OSError(f"Incomplete download: got {actual:,} of {total:,} expected bytes for {path.name}")


def _is_dns_error(error: Exception) -> bool:
    return isinstance(error, socket.gaierror) or any(marker in str(error) for marker in DNS_ERROR_MARKERS)


def download(category: str, retries: int | None = None, backoff: float = RETRY_BACKOFF_SECONDS) -> Path:
    dest = manual_download_path(category)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using already-downloaded file for {category}: {dest}", flush=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = RESOLVE_URL.format(category=category)
    attempt = 0
    while True:
        attempt += 1
        try:
            _fetch_to_file(url, part, part.stat().st_size if part.exists() else 0)
            part.rename(dest)
            return dest
        except urllib.error.HTTPError as error:
            if error.code in FATAL_HTTP_STATUSES:
                raise RuntimeError(f"Download of {category} failed with HTTP {error.code} {error.reason} - not retrying") from error
            last_error = error
        except Exception as error:  # noqa: BLE001
            last_error = error
        if retries is not None and attempt >= retries:
            raise RuntimeError(f"Failed to download {category} after {attempt} attempts") from last_error
        wait = DNS_RETRY_BACKOFF_SECONDS if _is_dns_error(last_error) else min(backoff * attempt, MAX_BACKOFF_SECONDS)
        done = part.stat().st_size if part.exists() else 0
        print(f"Download of {category} failed on attempt {attempt} at {done / 1e6:.0f}MB ({last_error}); retrying in {wait:.0f}s...", flush=True)
        time.sleep(wait)


def parse_chunk(lines: list[str], category: str) -> list[Item]:
    items = (parse(json.loads(line), category) for line in lines if line.strip())
    return [item for item in items if item]


def read_chunks(path: Path, size: int = CHUNK_SIZE):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        while chunk := list(islice(f, size)):
            yield chunk


def load_category(category: str, workers: int = WORKERS, path: Path | None = None) -> list[Item]:
    start = datetime.now()
    path = path or download(category)
    results = []
    chunks = read_chunks(path)
    with ProcessPoolExecutor(max_workers=workers) as pool, tqdm(desc=category, unit=" rows") as bar:
        while window := list(islice(chunks, workers * 4)):
            for batch in pool.map(parse_chunk, window, [category] * len(window)):
                results.extend(batch)
            bar.update(sum(len(chunk) for chunk in window))
    minutes = (datetime.now() - start).total_seconds() / 60
    print(f"Completed {category} with {len(results):,} items in {minutes:.1f} mins", flush=True)
    return results


def cache_path(category: str) -> Path:
    return config.DATA_DIR / "raw_cache" / f"{category}.jsonl"


def load_cached_category(category: str, workers: int = WORKERS, force: bool = False) -> list[Item]:
    path = cache_path(category)
    if path.exists() and not force:
        with path.open(encoding="utf-8") as f:
            items = [Item.model_validate_json(line) for line in f if line.strip()]
        print(f"Using {len(items):,} cached {category} items from {path}", flush=True)
        return items
    items = load_category(category, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.writelines(item.model_dump_json() + "\n" for item in items)
    temp.replace(path)
    print(f"Cached {len(items):,} {category} items to {path}", flush=True)
    return items