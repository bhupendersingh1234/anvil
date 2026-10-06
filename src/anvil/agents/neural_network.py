from pathlib import Path

from anvil import config
from anvil.agents.base import Agent
from anvil.log import MAGENTA
from anvil.models.deep_neural_network import DeepNeuralNetworkPricer


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = MAGENTA

    def __init__(self, weights: Path = config.DNN_WEIGHTS):
        self.log("Neural Network Agent is initializing")
        self.neural_network = DeepNeuralNetworkPricer()
        self.neural_network.load(weights)
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str) -> float:
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.price(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
