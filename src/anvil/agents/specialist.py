import modal

from anvil import config
from anvil.agents.base import Agent
from anvil.log import RED


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = RED

    def __init__(self):
        self.log("Specialist Agent is initializing - connecting to modal")
        self.service = modal.Cls.from_name(config.MODAL_APP, "Anvil")()

    def price(self, description: str) -> float:
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.service.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
