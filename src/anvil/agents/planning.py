from anvil import config
from anvil.agents.base import Agent
from anvil.agents.deals import Deal, Opportunity
from anvil.agents.ensemble import EnsembleAgent
from anvil.agents.messaging import MessagingAgent
from anvil.agents.scanner import ScannerAgent
from anvil.log import GREEN


class PlanningAgent(Agent):
    name = "Planning Agent"
    color = GREEN

    def __init__(self, collection, threshold: float = config.DEAL_THRESHOLD):
        self.log("Planning Agent is initializing")
        self.threshold = threshold
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: list[Opportunity] | None = None) -> Opportunity | None:
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory or [])
        if not selection or not selection.deals:
            return None
        best = max((self.run(deal) for deal in selection.deals[:5]), key=lambda opportunity: opportunity.discount)
        self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
        if best.discount <= self.threshold:
            self.log("Planning Agent has completed a run")
            return None
        self.messenger.alert(best)
        self.log("Planning Agent has completed a run")
        return best
