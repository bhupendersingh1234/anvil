from openai import OpenAI

from anvil import config, rag
from anvil.agents.base import Agent
from anvil.data.items import extract_price
from anvil.log import BLUE


class FrontierAgent(Agent):
    name = "Frontier Agent"
    color = BLUE

    def __init__(self, collection, encoder=None, model: str = config.FRONTIER_MODEL):
        self.log("Initializing Frontier Agent")
        self.client = OpenAI()
        self.model = model
        self.collection = collection
        self.encoder = encoder or rag.load_encoder()
        self.log("Frontier Agent is ready")

    def price(self, description: str) -> float:
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        similars, prices = rag.find_similars(self.collection, self.encoder, description)
        self.log(f"Frontier Agent is about to call {self.model} with context including 5 similar products")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=rag.messages_for(description, similars, prices),
            seed=42,
            **({"reasoning_effort": "none"} if self.model.startswith(("gpt-5", "o")) else {}),
        )
        result = extract_price(response.choices[0].message.content)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result