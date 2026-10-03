from anvil.data.items import Item, extract_price, inference_prompt


def test_make_prompt_sets_prompt_with_rounded_price() -> None:
    item = Item(title="Widget", category="Tools", price=19.6)
    item.make_prompt("A great widget")
    assert item.prompt == "What does this cost to the nearest dollar?\n\nA great widget\n\nPrice is $20.00"


def test_test_prompt_strips_price() -> None:
    item = Item(title="Widget", category="Tools", price=19.6)
    item.make_prompt("A great widget")
    assert item.test_prompt() == inference_prompt("A great widget")


def test_repr() -> None:
    assert repr(Item(title="Widget", category="Tools", price=19.6)) == "<Widget = $19.6>"


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def test_make_prompts_truncates_and_rounds() -> None:
    item = Item(title="Widget", category="Tools", price=19.6, summary="one two three four five")
    item.make_prompts(FakeTokenizer(), 3, True)
    assert item.prompt == inference_prompt("one two three")
    assert item.completion == "20.00"
    assert item.to_datapoint() == {"prompt": item.prompt, "completion": "20.00"}
    item.make_prompts(FakeTokenizer(), 10, False)
    assert item.completion == "19.6"
    assert item.count_prompt_tokens(FakeTokenizer()) == len((item.prompt + item.completion).split())


def test_extract_price() -> None:
    assert extract_price("$1,234.50") == 1234.5
    assert extract_price("It costs about 42 dollars") == 42
    assert extract_price("no idea") == 0
    assert extract_price(7) == 7.0
