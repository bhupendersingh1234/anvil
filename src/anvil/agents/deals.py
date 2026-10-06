import logging
import re
import time
from typing import Self

import feedparser
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
]
ENTRIES_PER_FEED = 10
TIMEOUT = 15


def extract(html_snippet: str) -> str:
    snippet = BeautifulSoup(html_snippet, "html.parser").find("div", class_="snippet summary")
    if not snippet:
        return html_snippet.replace("\n", " ")
    text = BeautifulSoup(snippet.get_text(" ", strip=True), "html.parser").get_text()
    return re.sub("<[^<]+?>", "", text).strip().replace("\n", " ")


class ScrapedDeal:
    def __init__(self, title: str, summary: str, url: str, content: str):
        self.title = title[:100]
        self.summary = summary
        self.url = url
        details, _, features = content.replace("\nmore", "").replace("\n", " ").partition("Features")
        self.details = details[:500]
        self.features = features[:500]

    @classmethod
    def from_entry(cls, entry: dict) -> Self:
        url = entry["links"][0]["href"]
        page = BeautifulSoup(requests.get(url, timeout=TIMEOUT).content, "html.parser")
        section = page.find("div", class_="content-section")
        return cls(entry["title"], extract(entry["summary"]), url, section.get_text() if section else "")

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self) -> str:
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, feeds: list[str] = FEEDS) -> list[Self]:
        deals = []
        for feed_url in feeds:
            for entry in feedparser.parse(feed_url).entries[:ENTRIES_PER_FEED]:
                try:
                    deals.append(cls.from_entry(entry))
                except (requests.RequestException, KeyError, IndexError) as error:
                    logging.warning(f"Skipping deal {entry.get('title', '?')}: {error}")
                time.sleep(0.05)
        return deals


class Deal(BaseModel):
    product_description: str = Field(
        description="Your clearly expressed summary of the product in 3-4 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a short paragraph of text for each item you choose."
    )
    price: float = Field(
        description="The actual price of this product, as advertised in the deal. Be sure to give the actual price; for example, if a deal is described as $100 off the usual $300 price, you should respond with $200"
    )
    url: str = Field(description="The URL of the deal, as provided in the input")


class DealSelection(BaseModel):
    deals: list[Deal] = Field(
        description="Your selection of the 5 deals that have the most detailed, high quality description and the most clear price. You should be confident that the price reflects the deal, that it is a good deal, with a clear description"
    )


class Opportunity(BaseModel):
    deal: Deal
    estimate: float
    discount: float
