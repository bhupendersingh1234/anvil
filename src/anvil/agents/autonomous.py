import json

from openai import OpenAI

from anvil import config
from anvil.agents.base import Agent
from anvil.agents.deals import Deal, Opportunity
from anvil.agents.ensemble import EnsembleAgent
from anvil.agents.messaging import MessagingAgent
from anvil.agents.scanner import ScannerAgent
from anvil.log import GREEN

SYSTEM_MESSAGE = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
USER_MESSAGE = """First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
Then just reply OK to indicate success."""
MAX_TURNS = 20


def function(name: str, description: str, properties: dict) -> dict:
    parameters = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [
    function("scan_the_internet_for_bargains", "Returns top bargains scraped from the internet along with the price each item is being offered for", {}),
    function(
        "estimate_true_value",
        "Given the description of an item, estimate how much it is actually worth",
        {
            "description": {"type": "string", "description": "The description of the item to be estimated"},
        },
    ),
    function(
        "notify_user_of_deal",
        "Send the user a push notification about the single most compelling deal; only call this one time",
        {
            "description": {"type": "string", "description": "The description of the item itself scraped from the internet"},
            "deal_price": {"type": "number", "description": "The price offered by this deal scraped from the internet"},
            "estimated_true_value": {"type": "number", "description": "The estimated actual value that this is worth"},
            "url": {"type": "string", "description": "The URL of this deal as scraped from the internet"},
        },
    ),
]


class AutonomousPlanningAgent(Agent):
    name = "Autonomous Planning Agent"
    color = GREEN

    def __init__(self, collection, model: str = config.PLANNER_MODEL):
        self.log("Autonomous Planning Agent is initializing")
        self.model = model
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory: list[Opportunity] = []
        self.opportunity: Opportunity | None = None
        self.log("Autonomous Planning Agent is ready")

    def scan_the_internet_for_bargains(self) -> str:
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        return f"The estimated true value of {description} is {self.ensemble.price(description)}"

    def notify_user_of_deal(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
            return "Notification already sent"
        self.log("Autonomous Planning agent is notifying user")
        self.messenger.notify(description, deal_price, estimated_true_value, url)
        deal = Deal(product_description=description, price=deal_price, url=url)
        self.opportunity = Opportunity(deal=deal, estimate=estimated_true_value, discount=estimated_true_value - deal_price)
        return "Notification sent ok"

    def handle_tool_calls(self, message) -> list[dict]:
        tools = {tool["function"]["name"]: getattr(self, tool["function"]["name"]) for tool in TOOLS}
        results = []
        for call in message.tool_calls:
            tool = tools.get(call.function.name)
            result = tool(**json.loads(call.function.arguments)) if tool else f"Unknown tool {call.function.name}"
            results.append({"role": "tool", "content": result, "tool_call_id": call.id})
        return results

    def plan(self, memory: list[Opportunity] | None = None) -> Opportunity | None:
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory or []
        self.opportunity = None
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": USER_MESSAGE}]
        for _ in range(MAX_TURNS):
            response = self.openai.chat.completions.create(model=self.model, messages=messages, tools=TOOLS)
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                self.log(f"Autonomous Planning Agent completed with: {choice.message.content}")
                break
            messages.append(choice.message)
            messages.extend(self.handle_tool_calls(choice.message))
        return self.opportunity
