from litellm import completion

from anvil.data.items import Item
from anvil.models.baselines import Predictor

INSTRUCTION = "Estimate the price of this product. Respond with the price, no explanation"


def messages_for(text: str) -> list[dict]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]


def frontier_pricer(model: str, **kwargs) -> Predictor:
    def predict(item: Item) -> str:
        return completion(model=model, messages=messages_for(item.summary), **kwargs).choices[0].message.content

    return predict
