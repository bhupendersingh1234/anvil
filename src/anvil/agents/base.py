import logging

from anvil.log import BG_BLACK, RESET, WHITE


class Agent:
    name: str = ""
    color: str = WHITE

    def log(self, message: str) -> None:
        logging.info(f"{BG_BLACK}{self.color}[{self.name}] {message}{RESET}")
