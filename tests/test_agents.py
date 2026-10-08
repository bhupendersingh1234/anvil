import json
import logging
from types import SimpleNamespace

import pytest

from anvil import config
from anvil.agents import deals as deals_module
from anvil.agents.autonomous import TOOLS, AutonomousPlanningAgent
from anvil.agents.deals import Deal, DealSelection, Opportunity, ScrapedDeal, extract
from anvil.agents.ensemble import EnsembleAgent
from anvil.agents.framework import DealAgentFramework, read_memory, reset_memory, write_memory
from anvil.agents.messaging import MessagingAgent, alert_text
from anvil.agents.planning import PlanningAgent
from anvil.agents.scanner import ScannerAgent, make_user_prompt
from anvil.log import BG_BLACK, RED, RESET, to_html


def opportunity(url: str, price: float = 100, estimate: float = 300) -> Opportunity:
    return Opportunity(deal=Deal(product_description="A fine TV", price=price, url=url), estimate=estimate, discount=estimate - price)


def build(cls, **attributes):
    agent = cls.__new__(cls)
    agent.__dict__.update(attributes)
    return agent


class Fixed:
    def __init__(self, value: float):
        self.value = value

    def price(self, description: str) -> float:
        return self.value


class Recorder:
    def __init__(self):
        self.calls = []

    def alert(self, opp) -> None:
        self.calls.append(("alert", opp))

    def notify(self, *args) -> None:
        self.calls.append(("notify", args))


def test_extract_cleans_snippet() -> None:
    assert extract('<div class="snippet summary">Great <b>TV</b>\n deal</div>') == "Great TV deal"
    assert extract("plain\ntext") == "plain text"


def test_scraped_deal_splits_features_and_truncates() -> None:
    deal = ScrapedDeal("T" * 150, "summary", "http://x", "Details here\nmore Features fast and small")
    assert len(deal.title) == 100 and deal.details.strip() == "Details here" and deal.features.strip() == "fast and small"
    assert "URL: http://x" in deal.describe()


def test_scraped_deal_fetch_skips_broken_pages(monkeypatch) -> None:
    entries = [{"title": "ok", "summary": "s", "links": [{"href": "http://ok"}]}, {"title": "broken", "summary": "s", "links": []}]
    monkeypatch.setattr(deals_module.feedparser, "parse", lambda url: SimpleNamespace(entries=entries))
    monkeypatch.setattr(deals_module.requests, "get", lambda url, timeout: SimpleNamespace(content=b'<div class="content-section">Body</div>'))
    monkeypatch.setattr(deals_module.time, "sleep", lambda s: None)
    deals = ScrapedDeal.fetch(feeds=["feed"])
    assert [deal.url for deal in deals] == ["http://ok"]


def test_scanner_filters_deals_already_in_memory(monkeypatch) -> None:
    scraped = [ScrapedDeal("a", "", "http://a", ""), ScrapedDeal("b", "", "http://b", "")]
    monkeypatch.setattr(ScrapedDeal, "fetch", classmethod(lambda cls: scraped))
    scanner = build(ScannerAgent)
    assert [deal.url for deal in scanner.fetch_deals([opportunity("http://a")])] == ["http://b"]
    assert "Include exactly 5 deals" in make_user_prompt(scraped)
    assert len(scanner.test_scan().deals) == 4


def test_planning_alerts_best_deal_above_threshold() -> None:
    selection = DealSelection(deals=[Deal(product_description="x", price=p, url=f"http://{p}") for p in (250, 100, 280)])
    messenger = Recorder()
    planner = build(PlanningAgent, threshold=50, scanner=SimpleNamespace(scan=lambda memory: selection), ensemble=Fixed(300), messenger=messenger)
    best = planner.plan()
    assert best.deal.price == 100 and best.discount == 200
    assert messenger.calls == [("alert", best)]


def test_planning_returns_none_below_threshold() -> None:
    selection = DealSelection(deals=[Deal(product_description="x", price=290, url="http://a")])
    messenger = Recorder()
    planner = build(PlanningAgent, threshold=50, scanner=SimpleNamespace(scan=lambda memory: selection), ensemble=Fixed(300), messenger=messenger)
    assert planner.plan() is None and messenger.calls == []


def test_ensemble_combines_weighted_estimates() -> None:
    agent = build(
        EnsembleAgent,
        weights=config.ENSEMBLE_WEIGHTS,
        preprocessor=SimpleNamespace(preprocess=lambda text: text, model_name="fake"),
        frontier=Fixed(100),
        specialist=Fixed(200),
        neural_network=Fixed(300),
    )
    assert agent.price("thing") == pytest.approx(0.8 * 100 + 0.1 * 200 + 0.1 * 300)


def test_ensemble_without_specialist_clamps_to_price_range() -> None:
    agent = build(
        EnsembleAgent,
        weights={"frontier": 0.5, "neural_network": 0.5},
        preprocessor=SimpleNamespace(preprocess=lambda text: text, model_name="fake"),
        frontier=Fixed(5000),
        specialist=None,
        neural_network=Fixed(-20),
    )
    assert agent.price("thing") == pytest.approx(0.5 * 999.49 + 0.5 * 0.5)


def test_active_ensemble_weights(monkeypatch) -> None:
    monkeypatch.setattr(config, "USE_SPECIALIST", False)
    assert config.active_ensemble_weights() == pytest.approx({"frontier": 0.8 / 0.9, "neural_network": 0.1 / 0.9})
    monkeypatch.setattr(config, "USE_SPECIALIST", True)
    assert config.active_ensemble_weights() == pytest.approx(config.ENSEMBLE_WEIGHTS)


def tool_call(name: str, arguments: dict, call_id: str):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def test_autonomous_agent_tool_loop() -> None:
    selection = DealSelection(deals=[Deal(product_description="TV", price=100, url="http://tv")])
    turns = iter(
        [
            [tool_call("scan_the_internet_for_bargains", {}, "1")],
            [tool_call("estimate_true_value", {"description": "TV"}, "2")],
            [tool_call("notify_user_of_deal", {"description": "TV", "deal_price": 100, "estimated_true_value": 300, "url": "http://tv"}, "3")] * 2,
            None,
        ]
    )

    def create(model, messages, tools):
        calls = next(turns)
        message = SimpleNamespace(tool_calls=calls, content="OK")
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls" if calls else "stop", message=message)])

    messenger = Recorder()
    agent = build(
        AutonomousPlanningAgent,
        model="fake",
        scanner=SimpleNamespace(scan=lambda memory: selection),
        ensemble=Fixed(300),
        messenger=messenger,
        openai=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result = agent.plan()
    assert result.discount == 200 and len(messenger.calls) == 1
    assert [tool["function"]["name"] for tool in TOOLS] == ["scan_the_internet_for_bargains", "estimate_true_value", "notify_user_of_deal"]


def test_messaging_without_credentials_only_logs(monkeypatch, caplog) -> None:
    monkeypatch.delenv("PUSHOVER_USER", raising=False)
    monkeypatch.delenv("PUSHOVER_TOKEN", raising=False)
    monkeypatch.setattr("anvil.agents.messaging.requests.post", lambda *a, **k: pytest.fail("should not post"))
    with caplog.at_level(logging.INFO):
        MessagingAgent().alert(opportunity("http://tv"))
    assert "Deal Alert! Price=$100.00, Estimate=$300.00, Discount=$200.00" in alert_text(opportunity("http://tv"))
    assert "push notification" in caplog.text


def test_memory_roundtrip_and_reset(tmp_path) -> None:
    path = tmp_path / "memory.json"
    assert read_memory(path) == []
    write_memory(path, [opportunity(f"http://{i}") for i in range(4)])
    assert len(read_memory(path)) == 4
    reset_memory(path)
    assert [o.deal.url for o in read_memory(path)] == ["http://0", "http://1"]


def test_framework_run_appends_to_memory(tmp_path) -> None:
    framework = DealAgentFramework(memory_file=tmp_path / "memory.json", vectorstore=tmp_path / "store")
    framework.planner = SimpleNamespace(plan=lambda memory: opportunity("http://new"))
    assert [o.deal.url for o in framework.run()] == ["http://new"]
    assert read_memory(tmp_path / "memory.json")[0].deal.url == "http://new"


def test_log_to_html() -> None:
    assert to_html(f"{BG_BLACK}{RED}hello{RESET}") == '<span style="color: #dd0000">hello</span>'