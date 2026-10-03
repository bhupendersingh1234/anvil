import logging
import sys

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
BG_BLACK = "\033[40m"
BG_BLUE = "\033[44m"
RESET = "\033[0m"

HTML_COLORS = {
    BG_BLACK + RED: "#dd0000",
    BG_BLACK + GREEN: "#00dd00",
    BG_BLACK + YELLOW: "#dddd00",
    BG_BLACK + BLUE: "#0000ee",
    BG_BLACK + MAGENTA: "#aa00dd",
    BG_BLACK + CYAN: "#00dddd",
    BG_BLACK + WHITE: "#87CEEB",
    BG_BLUE + WHITE: "#ff7800",
}


def init_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if any(getattr(handler, "anvil", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.anvil = True
    handler.setFormatter(logging.Formatter("[%(asctime)s] [Agents] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))
    root.addHandler(handler)


def to_html(message: str) -> str:
    for code, color in HTML_COLORS.items():
        message = message.replace(code, f'<span style="color: {color}">')
    return message.replace(RESET, "</span>")
