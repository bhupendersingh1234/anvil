import logging
import queue
import threading
import time

import gradio as gr
import plotly.graph_objects as go

from anvil.agents.deals import Opportunity
from anvil.agents.framework import DealAgentFramework
from anvil.log import to_html

REFRESH_SECONDS = 300
TITLE = '<div style="text-align: center;font-size:24px"><strong>Anvil</strong> - Autonomous Agent Framework that hunts for deals</div>'
SUBTITLE = '<div style="text-align: center;font-size:14px">A RAG pipeline with a frontier model and a deep neural network collaborate to price live deals and send push notifications for the best ones.</div>'


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def html_for(log_data: list[str]) -> str:
    output = "<br>".join(log_data[-18:])
    return f'<div style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">{output}</div>'


def table_for(opportunities: list[Opportunity]) -> list[list[str]]:
    return [[o.deal.product_description, f"${o.deal.price:.2f}", f"${o.estimate:.2f}", f"${o.discount:.2f}", o.deal.url] for o in opportunities]


def plot_for(framework: DealAgentFramework, max_datapoints: int = 800) -> go.Figure:
    fig = go.Figure()
    if framework.collection.count() > 10:
        _, vectors, colors = framework.plot_data(max_datapoints)
        fig.add_trace(go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2], mode="markers", marker=dict(size=2, color=colors, opacity=0.7)))
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=2.2, z=1),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        height=400,
        margin=dict(r=5, b=1, l=5, t=2),
    )
    return fig


class App:
    def __init__(self, autonomous: bool = False):
        self.autonomous = autonomous
        self.framework: DealAgentFramework | None = None

    def get_framework(self) -> DealAgentFramework:
        if not self.framework:
            self.framework = DealAgentFramework(autonomous=self.autonomous)
        return self.framework

    def run_with_logging(self, log_data: list[str]):
        log_queue, result_queue = queue.Queue(), queue.Queue()
        handler = QueueHandler(log_queue)
        logging.getLogger().addHandler(handler)
        threading.Thread(target=lambda: result_queue.put(table_for(self.get_framework().run())), daemon=True).start()
        initial, final = table_for(self.get_framework().memory), None
        try:
            while final is None or not log_queue.empty():
                try:
                    log_data.append(to_html(log_queue.get_nowait()))
                except queue.Empty:
                    try:
                        final = result_queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.1)
                        continue
                yield log_data, html_for(log_data), final or initial
        finally:
            logging.getLogger().removeHandler(handler)

    def select(self, selected: gr.SelectData) -> None:
        framework = self.get_framework()
        framework.init_agents_as_needed()
        framework.planner.messenger.alert(framework.memory[selected.index[0]])

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Anvil", fill_width=True) as ui:
            log_data = gr.State([])
            gr.Markdown(TITLE)
            gr.Markdown(SUBTITLE)
            table = gr.Dataframe(
                headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                wrap=True,
                column_widths=[6, 1, 1, 1, 3],
                row_count=10,
                col_count=5,
                max_height=400,
            )
            with gr.Row():
                logs = gr.HTML()
                gr.Plot(value=lambda: plot_for(self.get_framework()), show_label=False)
            ui.load(self.run_with_logging, inputs=[log_data], outputs=[log_data, logs, table])
            gr.Timer(value=REFRESH_SECONDS, active=True).tick(self.run_with_logging, inputs=[log_data], outputs=[log_data, logs, table])
            table.select(self.select)
        return ui

    def launch(self, share: bool = False) -> None:
        self.build().launch(share=share, inbrowser=True)