from openai import OpenAI

from anvil import config
from anvil.agents.base import Agent
from anvil.agents.deals import DealSelection, Opportunity, ScrapedDeal
from anvil.log import CYAN

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.
"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price.

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

TEST_DEALS = {
    "deals": [
        {
            "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers stunning picture quality with 3840x2160 resolution. It features Dolby Vision HDR and HDR10 compatibility, ensuring a vibrant and dynamic viewing experience. The TV runs on Roku's operating system, allowing easy access to streaming services and voice control compatibility with Google Assistant and Alexa. With three HDMI ports available, connecting multiple devices is simple and efficient.",
            "price": 178,
            "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
        },
        {
            "product_description": "The Poly Studio P21 is a 21.5-inch LED personal meeting display designed specifically for remote work and video conferencing. With a native resolution of 1080p, it provides crystal-clear video quality, featuring a privacy shutter and stereo speakers. This display includes a 1080p webcam with manual pan, tilt, and zoom control, along with an ambient light sensor to adjust the vanity lighting as needed. It also supports 5W wireless charging for mobile devices, making it an all-in-one solution for home offices.",
            "price": 30,
            "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
        },
        {
            "product_description": "The Lenovo IdeaPad Slim 5 laptop is powered by a 7th generation AMD Ryzen 5 8645HS 6-core CPU, offering efficient performance for multitasking and demanding applications. It features a 16-inch touch display with a resolution of 1920x1080, ensuring bright and vivid visuals. Accompanied by 16GB of RAM and a 512GB SSD, the laptop provides ample speed and storage for all your files. This model is designed to handle everyday tasks with ease while delivering an enjoyable user experience.",
            "price": 446,
            "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
        },
        {
            "product_description": "The Dell G15 gaming laptop is equipped with a 6th-generation AMD Ryzen 5 7640HS 6-Core CPU, providing powerful performance for gaming and content creation. It features a 15.6-inch 1080p display with a 120Hz refresh rate, allowing for smooth and responsive gameplay. With 16GB of RAM and a substantial 1TB NVMe M.2 SSD, this laptop ensures speedy performance and plenty of storage for games and applications. Additionally, it includes the Nvidia GeForce RTX 3050 GPU for enhanced graphics and gaming experiences.",
            "price": 650,
            "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
        },
    ]
}


def make_user_prompt(scraped: list[ScrapedDeal]) -> str:
    return USER_PROMPT_PREFIX + "\n\n".join(scrape.describe() for scrape in scraped) + USER_PROMPT_SUFFIX


class ScannerAgent(Agent):
    name = "Scanner Agent"
    color = CYAN

    def __init__(self, model: str = config.SCANNER_MODEL):
        self.log("Scanner Agent is initializing")
        self.model = model
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory: list[Opportunity]) -> list[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        seen = {opportunity.deal.url for opportunity in memory}
        result = [scrape for scrape in ScrapedDeal.fetch() if scrape.url not in seen]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def scan(self, memory: list[Opportunity] | None = None) -> DealSelection | None:
        scraped = self.fetch_deals(memory or [])
        if not scraped:
            return None
        self.log("Scanner Agent is calling OpenAI using Structured Outputs")
        response = self.openai.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": make_user_prompt(scraped)}],
            response_format=DealSelection,
            **({"reasoning_effort": "minimal"} if self.model.startswith(("gpt-5", "o")) else {}),
        )
        result = response.choices[0].message.parsed
        result.deals = [deal for deal in result.deals if deal.price > 0]
        self.log(f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI")
        return result

    def test_scan(self, memory: list[Opportunity] | None = None) -> DealSelection:
        return DealSelection(**TEST_DEALS)