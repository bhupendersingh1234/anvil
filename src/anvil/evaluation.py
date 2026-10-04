import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import accumulate

from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from anvil.data.items import extract_price

ANSI = {"red": "\033[91m", "orange": "\033[93m", "green": "\033[92m"}
RESET = "\033[0m"
WORKERS = 5
DEFAULT_SIZE = 200


@dataclass
class Result:
    title: str
    guess: float
    truth: float
    error: float
    color: str


@dataclass
class Metrics:
    title: str
    error: float
    ci: float
    mse: float
    r2: float

    def __str__(self) -> str:
        return f"{self.title}: error=${self.error:,.2f} ±${self.ci:,.2f} mse={self.mse:,.0f} r²={self.r2:.1f}%"


def make_title(predictor: Callable) -> str:
    return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")


def color_for(error: float, truth: float) -> str:
    if error < 40 or error / truth < 0.2:
        return "green"
    if error < 80 or error / truth < 0.4:
        return "orange"
    return "red"


def truth_and_title(datapoint) -> tuple[float, str]:
    if isinstance(datapoint, dict):
        pieces = datapoint["prompt"].split("Title: ")
        truth, title = float(datapoint["completion"]), pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
    else:
        truth, title = datapoint.price, datapoint.title
    return truth, title if len(title) <= 40 else title[:40] + "..."


def running_error(errors: list[float]) -> tuple[list[float], list[float]]:
    counts = range(1, len(errors) + 1)
    means = [s / i for s, i in zip(accumulate(errors), counts)]
    squares = accumulate(e * e for e in errors)
    stds = [math.sqrt(max(sq / i - m**2, 0)) if i > 1 else 0 for i, sq, m in zip(counts, squares, means)]
    return means, [1.96 * sd / math.sqrt(i) if i > 1 else 0 for i, sd in zip(counts, stds)]


class Tester:
    __test__ = False

    def __init__(self, predictor: Callable, data: list, title: str | None = None, size: int = DEFAULT_SIZE, workers: int = WORKERS):
        self.predictor = predictor
        self.data = data
        self.title = title or make_title(predictor)
        self.size = min(size, len(data))
        self.workers = workers
        self.results: list[Result] = []

    def run_datapoint(self, i: int) -> Result:
        datapoint = self.data[i]
        guess = extract_price(self.predictor(datapoint))
        truth, title = truth_and_title(datapoint)
        error = abs(guess - truth)
        return Result(title, guess, truth, error, color_for(error, truth))

    def metrics(self) -> Metrics:
        errors = [r.error for r in self.results]
        truths, guesses = [r.truth for r in self.results], [r.guess for r in self.results]
        means, ci = running_error(errors)
        return Metrics(self.title, means[-1], ci[-1], mean_squared_error(truths, guesses), r2_score(truths, guesses) * 100)

    def run(self, show: bool = False, verbose: bool = True) -> Metrics:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for result in tqdm(pool.map(self.run_datapoint, range(self.size)), total=self.size, disable=not verbose):
                self.results.append(result)
                if verbose:
                    print(f"{ANSI[result.color]}${result.error:.0f} {RESET}", end="")
        metrics = self.metrics()
        if verbose:
            print(f"\n{metrics}")
        if show:
            self.error_trend_chart(metrics).show()
            self.scatter_chart(metrics).show()
        return metrics

    def scatter_chart(self, metrics: Metrics):
        import plotly.graph_objects as go

        max_val = max(max(r.truth for r in self.results), max(r.guess for r in self.results))
        fig = go.Figure()
        for color in ("green", "orange", "red"):
            rows = [r for r in self.results if r.color == color]
            fig.add_trace(
                go.Scatter(
                    x=[r.truth for r in rows],
                    y=[r.guess for r in rows],
                    mode="markers",
                    marker=dict(size=6, color=color),
                    customdata=[f"{r.title}<br>Guess=${r.guess:,.2f} Actual=${r.truth:,.2f}" for r in rows],
                    hovertemplate="%{customdata}<extra></extra>",
                )
            )
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(width=2, dash="dash", color="deepskyblue"), hoverinfo="skip"))
        title = f"{metrics.title} results<br><b>Error:</b> ${metrics.error:,.2f} <b>MSE:</b> {metrics.mse:,.0f} <b>r²:</b> {metrics.r2:.1f}%"
        fig.update_layout(title=title, xaxis_title="Actual Price", yaxis_title="Predicted Price", width=1000, height=800, showlegend=False)
        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        return fig

    def error_trend_chart(self, metrics: Metrics):
        import plotly.graph_objects as go

        means, ci = running_error([r.error for r in self.results])
        x = list(range(1, len(means) + 1))
        upper, lower = [m + c for m, c in zip(means, ci)], [m - c for m, c in zip(means, ci)]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(128,128,128,0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                customdata=[[c] for c in ci],
                hovertemplate="n=%{x}<br>Avg Error=$%{y:,.2f}<br>±95% CI=$%{customdata[0]:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"{metrics.title} Error: ${metrics.error:,.2f} ± ${metrics.ci:,.2f}",
            xaxis_title="Number of Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )
        return fig


def evaluate(predictor: Callable, data: list, title: str | None = None, size: int = DEFAULT_SIZE, workers: int = WORKERS, show: bool = False) -> Metrics:
    return Tester(predictor, data, title=title, size=size, workers=workers).run(show=show)
