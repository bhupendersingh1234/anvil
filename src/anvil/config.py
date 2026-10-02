import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

DATA_DIR = Path(os.getenv("ANVIL_DATA_DIR", "data"))
VECTORSTORE_DIR = DATA_DIR / "products_vectorstore"
MEMORY_FILE = DATA_DIR / "memory.json"
DNN_WEIGHTS = DATA_DIR / "deep_neural_network.pth"
BATCH_DIR = DATA_DIR / "batches"
JSONL_DIR = DATA_DIR / "jsonl"
HUMAN_PREDICTIONS = Path("benchmarks/human_out.csv")

HF_SOURCE = os.getenv("ANVIL_HF_SOURCE", "bhupi1234")
HF_USER = os.getenv("ANVIL_HF_USER", "")

CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]

SPLITS = {True: (20_000, 1_000, 1_000), False: (800_000, 10_000, 10_000)}

PREPROCESSOR_MODEL = os.getenv("ANVIL_PREPROCESSOR_MODEL", "gpt-4.1-nano")
BATCH_MODEL = os.getenv("ANVIL_BATCH_MODEL", "openai/gpt-oss-20b")
FRONTIER_MODEL = os.getenv("ANVIL_FRONTIER_MODEL", "gpt-4.1-nano")
SCANNER_MODEL = os.getenv("ANVIL_SCANNER_MODEL", "gpt-4.1-nano")
PLANNER_MODEL = os.getenv("ANVIL_PLANNER_MODEL", "gpt-4.1-nano")
MESSAGING_MODEL = os.getenv("ANVIL_MESSAGING_MODEL", "gpt-4.1-nano")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = os.getenv("ANVIL_FINETUNED_MODEL", "bhupi1234/anvil-price")
FINETUNED_REVISION = os.getenv("ANVIL_FINETUNED_REVISION") or None
MODAL_APP = "anvil-service"

DEAL_THRESHOLD = float(os.getenv("ANVIL_DEAL_THRESHOLD", "50"))
USE_SPECIALIST = os.getenv("ANVIL_USE_SPECIALIST", "false").lower() == "true"
ENSEMBLE_WEIGHTS = {"frontier": 0.8, "specialist": 0.1, "neural_network": 0.1}


def active_ensemble_weights() -> dict[str, float]:
    weights = {name: value for name, value in ENSEMBLE_WEIGHTS.items() if USE_SPECIALIST or name != "specialist"}
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def dataset_name(kind: str, lite: bool, user: str = HF_SOURCE) -> str:
    return f"{user}/{kind}_{'lite' if lite else 'full'}"


def require_hf_user() -> str:
    if not HF_USER:
        raise RuntimeError("Set ANVIL_HF_USER in .env to your HuggingFace username before pushing to the Hub")
    return HF_USER


def push_target(kind: str, lite: bool) -> str:
    return dataset_name(kind, lite, require_hf_user())