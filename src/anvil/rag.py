from pathlib import Path

import chromadb
import numpy as np
from tqdm.auto import tqdm

from anvil import config
from anvil.data.items import Item

COLLECTION = "products"
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def get_collection(path: Path = config.VECTORSTORE_DIR):
    return chromadb.PersistentClient(path=str(path)).get_or_create_collection(COLLECTION)


def load_encoder(model: str = config.EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def build_index(collection, encoder, items: list[Item], batch_size: int = 1000) -> int:
    start = collection.count()
    for i in tqdm(range(start, len(items), batch_size), desc="Indexing"):
        batch = items[i : i + batch_size]
        documents = [item.summary for item in batch]
        collection.add(
            ids=[f"doc_{j}" for j in range(i, i + len(batch))],
            documents=documents,
            embeddings=encoder.encode(documents).astype(float).tolist(),
            metadatas=[{"category": item.category, "price": item.price} for item in batch],
        )
    return collection.count()


def find_similars(collection, encoder, text: str, n: int = 5) -> tuple[list[str], list[float]]:
    results = collection.query(query_embeddings=encoder.encode([text]).astype(float).tolist(), n_results=n)
    return results["documents"][0], [m["price"] for m in results["metadatas"][0]]


def make_context(similars: list[str], prices: list[float]) -> str:
    message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
    return message + "".join(f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n" for similar, price in zip(similars, prices))


def messages_for(text: str, similars: list[str], prices: list[float]) -> list[dict]:
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{text}\n\n{make_context(similars, prices)}"
    return [{"role": "user", "content": message}]


def rag_pricer(collection, encoder, model: str = config.FRONTIER_MODEL, **kwargs):
    from litellm import completion

    def predict(item: Item) -> str:
        similars, prices = find_similars(collection, encoder, item.summary)
        return completion(model=model, messages=messages_for(item.summary, similars, prices), **kwargs).choices[0].message.content

    return predict


def plot_data(collection, max_datapoints: int = 2000, dimensions: int = 3) -> tuple[list[str], np.ndarray, list[str]]:
    from sklearn.manifold import TSNE

    result = collection.get(include=["embeddings", "documents", "metadatas"], limit=max_datapoints)
    vectors = np.array(result["embeddings"])
    colors = [COLORS[config.CATEGORIES.index(m["category"])] for m in result["metadatas"]]
    reduced = TSNE(n_components=dimensions, random_state=42, perplexity=min(30, len(vectors) - 1)).fit_transform(vectors)
    return result["documents"], reduced, colors
