from anvil import config
from anvil.agents.base import Agent
from anvil.agents.frontier import FrontierAgent
from anvil.agents.neural_network import NeuralNetworkAgent
from anvil.data.parser import MAX_PRICE, MIN_PRICE
from anvil.data.summarize import Preprocessor
from anvil.log import YELLOW


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = YELLOW

    def __init__(self, collection):
        self.log("Initializing Ensemble Agent")
        self.weights = config.active_ensemble_weights()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.specialist = None
        if config.USE_SPECIALIST:
            from anvil.agents.specialist import SpecialistAgent

            self.specialist = SpecialistAgent()
        else:
            self.log("Specialist disabled (ANVIL_USE_SPECIALIST=false) - using Frontier + Neural Network")
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        estimates = {"frontier": self.frontier.price(rewrite), "neural_network": self.neural_network.price(rewrite)}
        if self.specialist:
            estimates["specialist"] = self.specialist.price(rewrite)
        combined = sum(self.weights[name] * min(max(value, MIN_PRICE), MAX_PRICE) for name, value in estimates.items())
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined