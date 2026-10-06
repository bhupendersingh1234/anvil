import os

import requests
from litellm import completion

from anvil import config
from anvil.agents.base import Agent
from anvil.agents.deals import Opportunity
from anvil.log import WHITE

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def alert_text(opportunity: Opportunity) -> str:
    return (
        f"Deal Alert! Price=${opportunity.deal.price:.2f}, Estimate=${opportunity.estimate:.2f}, "
        f"Discount=${opportunity.discount:.2f} :{opportunity.deal.product_description[:10]}... {opportunity.deal.url}"
    )


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = WHITE

    def __init__(self, model: str = config.MESSAGING_MODEL):
        self.log("Messaging Agent is initializing")
        self.model = model
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        if not (self.pushover_user and self.pushover_token):
            self.log("PUSHOVER_USER / PUSHOVER_TOKEN not set - notifications will only be logged")
        self.log("Messaging Agent has initialized Pushover")

    def push(self, text: str) -> None:
        self.log(f"Messaging Agent is sending a push notification: {text[:80]}")
        if self.pushover_user and self.pushover_token:
            payload = {"user": self.pushover_user, "token": self.pushover_token, "message": text, "sound": "cashregister"}
            requests.post(PUSHOVER_URL, data=payload, timeout=15).raise_for_status()

    def alert(self, opportunity: Opportunity) -> None:
        self.push(alert_text(opportunity))
        self.log("Messaging Agent has completed")

    def craft_message(self, description: str, deal_price: float, estimated_true_value: float) -> str:
        prompt = (
            "Please summarize this great deal in 2-3 sentences to be sent as an exciting push notification alerting the user about this deal.\n"
            f"Item Description: {description}\nOffered Price: {deal_price}\nEstimated true value: {estimated_true_value}"
            "\n\nRespond only with the 2-3 sentence message which will be used to alert & excite the user about this deal"
        )
        return completion(model=self.model, messages=[{"role": "user", "content": prompt}]).choices[0].message.content

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> None:
        self.log("Messaging Agent is using an LLM to craft the message")
        self.push(self.craft_message(description, deal_price, estimated_true_value)[:200] + "... " + url)
        self.log("Messaging Agent has completed")
