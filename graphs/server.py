"""Dash application — live training dashboard for TRM.

Six dynamically‑updating panels:
  1. Loss curve (train + val)
  2. Accuracy curve (train + val)
  3. Mean recursion depth over time
  4. Gate‑value histogram (latest step)
  5. Learning‑rate schedule
  6. Epoch wall‑time bar chart
"""

from __future__ import annotations

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from trm.utils import MetricsStore


COLORS = {
    "bg": "#0e1117",
    "card": "#1a1d23",
    "text": "#e0e0e0",
    "accent1": "#00d4ff",
    "accent2": "#ff6b6b",
    "accent3": "#51cf66",
    "accent4": "#ffd43b",
    "accent5": "#cc5de8",
    "accent6": "#ff922b",
    "grid": "#2a2d35",
}


def _base_layout(title: str) -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"]),
        font=dict(color=COLORS["text"], size=11),
    )


def create_app(metrics: MetricsStore) -> Dash:
    app = Dash(
        __name__,
        title="TRM — Live Training Dashboard",
        update_title=None,
    )

    app.layout = html.Div(
        style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "padding": "20px",
            "fontFamily": "monospace",
        },
        children=[
            html.H1(
                "🔄 Tiny Recursive Model — Live Graphs",
                style={"color": COLORS["accent1"], "textAlign": "center"},
            ),
            html.Div(
                id="status-bar",
                style={
                    "color": COLORS["text"],
                    "textAlign": "center",
                    "marginBottom": "20px",
                },
            ),
            # Row 1: Loss + Accuracy
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
                children=[
                    html.Div(dcc.Graph(id="loss-graph"), style={"flex": "1"}),
                    html.Div(dcc.Graph(id="acc-graph"), style={"flex": "1"}),
                ],
            ),
            # Row 2: Depth + Gate histogram
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
                children=[
                    html.Div(dcc.Graph(id="depth-graph"), style={"flex": "1"}),
                    html.Div(dcc.Graph(id="gate-hist"), style={"flex": "1"}),
                ],
            ),
            # Row 3: LR schedule + Epoch times
            html.Div(
                style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
                children=[
                    html.Div(dcc.Graph(id="lr-graph"), style={"flex": "1"}),
                    html.Div(dcc.Graph(id="epoch-time-graph"), style={"flex": "1"}),
                ],
            ),
            # Auto‑refresh every 2 s
            dcc.Interval(id="interval", interval=2000, n_intervals=0),
        ],
    )

    # ── callbacks ─────────────────────────────────────────────────────
    @app.callback(
        [
            Output("status-bar", "children"),
            Output("loss-graph", "figure"),
            Output("acc-graph", "figure"),
            Output("depth-graph", "figure"),
            Output("gate-hist", "figure"),
            Output("lr-graph", "figure"),
            Output("epoch-time-graph", "figure"),
        ],
        [Input("interval", "n_intervals")],
    )
    def update_graphs(_n: int):
        snap = metrics.snapshot()
        steps = snap["steps"]
        n_steps = len(steps)

        status = f"Step {steps[-1] if steps else 0}  ·  {n_steps} logged points"

        # 1 ── Loss
        loss_fig = go.Figure(layout=_base_layout("Training & Validation Loss"))
        if steps:
            loss_fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=snap["losses"],
                    mode="lines",
                    name="train",
                    line=dict(color=COLORS["accent1"]),
                )
            )
        if snap["val_steps"]:
            loss_fig.add_trace(
                go.Scatter(
                    x=snap["val_steps"],
                    y=snap["val_losses"],
                    mode="lines+markers",
                    name="val",
                    line=dict(color=COLORS["accent2"], dash="dash"),
                )
            )

        # 2 ── Accuracy
        acc_fig = go.Figure(layout=_base_layout("Training & Validation Accuracy"))
        if steps:
            acc_fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=snap["accuracies"],
                    mode="lines",
                    name="train",
                    line=dict(color=COLORS["accent3"]),
                )
            )
        if snap["val_steps"]:
            acc_fig.add_trace(
                go.Scatter(
                    x=snap["val_steps"],
                    y=snap["val_accuracies"],
                    mode="lines+markers",
                    name="val",
                    line=dict(color=COLORS["accent2"], dash="dash"),
                )
            )

        # 3 ── Mean Depth
        depth_fig = go.Figure(layout=_base_layout("Mean Recursion Depth"))
        if steps:
            depth_fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=snap["mean_depths"],
                    mode="lines",
                    name="depth",
                    line=dict(color=COLORS["accent4"]),
                    fill="tozeroy",
                    fillcolor="rgba(255,212,59,0.15)",
                )
            )

        # 4 ── Gate histogram (latest)
        gate_fig = go.Figure(layout=_base_layout("Gate Values (latest step)"))
        if snap["gate_histograms"]:
            latest_gates = snap["gate_histograms"][-1]
            gate_fig.add_trace(
                go.Histogram(
                    x=latest_gates,
                    nbinsx=30,
                    marker_color=COLORS["accent5"],
                    opacity=0.8,
                )
            )
            gate_fig.update_layout(xaxis_title="gate value", yaxis_title="count")

        # 5 ── Learning rate
        lr_fig = go.Figure(layout=_base_layout("Learning Rate Schedule"))
        if snap["learning_rates"]:
            lr_fig.add_trace(
                go.Scatter(
                    x=steps[: len(snap["learning_rates"])],
                    y=snap["learning_rates"],
                    mode="lines",
                    line=dict(color=COLORS["accent6"]),
                )
            )

        # 6 ── Epoch times
        epoch_fig = go.Figure(layout=_base_layout("Epoch Wall Time (s)"))
        if snap["epoch_times"]:
            epochs = list(range(1, len(snap["epoch_times"]) + 1))
            epoch_fig.add_trace(
                go.Bar(
                    x=epochs,
                    y=snap["epoch_times"],
                    marker_color=COLORS["accent1"],
                    opacity=0.8,
                )
            )
            epoch_fig.update_layout(xaxis_title="epoch", yaxis_title="seconds")

        return status, loss_fig, acc_fig, depth_fig, gate_fig, lr_fig, epoch_fig

    return app
