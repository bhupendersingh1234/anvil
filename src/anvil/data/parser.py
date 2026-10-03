import json
import re

from anvil.data.items import Item

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

CODE_PATTERN = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")

POUNDS_PER_UNIT = {"pounds": 1, "ounces": 1 / 16, "grams": 1 / 453.592, "milligrams": 1 / 453592, "kilograms": 1 / 0.453592}


def simplify(text_list) -> str:
    return str(text_list).replace("\n", " ").replace("\r", "").replace("\t", "").replace("  ", " ").strip()[:MAX_TEXT_EACH]


def scrub(title: str, description, features, details: dict) -> str:
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    return CODE_PATTERN.sub("", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details: dict) -> float:
    parts = str(details.get("Item Weight") or "").split(" ")
    try:
        amount = float(parts[0])
    except ValueError:
        return 0
    unit = parts[1].lower() if len(parts) > 1 else ""
    if unit in POUNDS_PER_UNIT:
        return amount * POUNDS_PER_UNIT[unit]
    if unit == "hundredths" and len(parts) > 2 and parts[2].lower() == "pounds":
        return amount / 100
    return 0


def parse(datapoint: dict, category: str) -> Item | None:
    try:
        price = float(datapoint["price"])
    except (TypeError, ValueError):
        return None
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None
    title = datapoint["title"] or ""
    details = datapoint.get("details") or {}
    details = json.loads(details) if isinstance(details, str) else dict(details)
    weight = get_weight(details)
    full = scrub(title, datapoint.get("description"), datapoint.get("features"), details)
    if len(full) < MIN_CHARS:
        return None
    return Item(title=title, category=category, price=price, full=full, weight=weight)
