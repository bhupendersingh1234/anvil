from conftest import raw_datapoint
from anvil.data.parser import get_weight, parse, scrub, simplify


def test_get_weight_converts_units() -> None:
    assert get_weight({"Item Weight": "2 pounds"}) == 2
    assert get_weight({"Item Weight": "16 ounces"}) == 1
    assert get_weight({"Item Weight": "250 hundredths pounds"}) == 2.5
    assert get_weight({"Item Weight": "heavy"}) == 0
    assert get_weight({}) == 0


def test_simplify_strips_whitespace() -> None:
    result = simplify("a\nb\tc")
    assert "\n" not in result and "\t" not in result


def test_scrub_removes_part_numbers_and_noise() -> None:
    result = scrub("Widget", ["desc"], [], {"Part Number": "X", "Model": "ABC1234567", "Color": "red"})
    assert "Part Number" not in result and "ABC1234567" not in result and "red" in result


def test_parse_accepts_hub_script_format() -> None:
    item = parse(raw_datapoint(1), "Appliances")
    assert item.price == 19.99 and item.weight == 2 and item.category == "Appliances"
    assert "ABC1234567" not in item.full


def test_parse_accepts_raw_jsonl_format() -> None:
    item = parse(raw_datapoint(1, price=19.99, details={"Item Weight": "32 ounces"}), "Appliances")
    assert item.weight == 2


def test_parse_rejects_bad_prices() -> None:
    for price in ("None", None, "1500.00", "0.10"):
        assert parse(raw_datapoint(1, price=price), "Test") is None


def test_parse_rejects_short_text() -> None:
    datapoint = {"price": "10.00", "title": "x", "description": "", "features": "", "details": "{}"}
    assert parse(datapoint, "Test") is None
