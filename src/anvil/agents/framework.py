import json
import logging
from pathlib import Path

from anvil import config, rag
from anvil.agents.deals import Opportunity
from anvil.log import BG_BLUE, RESET, WHITE, init_logging


def read_memory(path: Path) -> list[Opportunity]:
    return [Opportunity(**item) for item in json.loads(path.read_text())] if path.exists() else []


def write_memory(path: Path, memory: list[Opportunity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([opportunity.model_dump() for opportunity in memory], indent=2))


def reset_memory(path: Path = config.MEMORY_FILE, keep: int = 2) -> None:
    write_memory(path, read_memory(path)[:keep])


class DealAgentFramework:
    def __init__(self, memory_file: Path = config.MEMORY_FILE, vectorstore: Path = config.VECTORSTORE_DIR, autonomous: bool = False):
        init_logging()
        self.memory_file = memory_file
        self.memory = read_memory(memory_file)
        self.collection = rag.get_collection(vectorstore)
        self.autonomous = autonomous
        self.planner = None

    def log(self, message: str) -> None:
        logging.info(f"{BG_BLUE}{WHITE}[Agent Framework] {message}{RESET}")

    def init_agents_as_needed(self) -> None:
        if self.planner:
            return
        self.log("Initializing Agent Framework")
        if self.autonomous:
            from anvil.agents.autonomous import AutonomousPlanningAgent

            self.planner = AutonomousPlanningAgent(self.collection)
        else:
            from anvil.agents.planning import PlanningAgent

            self.planner = PlanningAgent(self.collection)
        self.log("Agent Framework is ready")

    def run(self) -> list[Opportunity]:
        self.init_agents_as_needed()
        self.log("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        self.log(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            write_memory(self.memory_file, self.memory)
        return self.memory

    def plot_data(self, max_datapoints: int = 2000):
        return rag.plot_data(self.collection, max_datapoints)
