import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


# =========================================================
# Build tag (чтобы ты видел, что запустился ИМЕННО этот файл)
# =========================================================
APP_BUILD = "UI_v4_uncertainty_quantiles__2025-12-24"


# =========================================================
# Helpers
# =========================================================
def first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def safe_load_json(path: str) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None



# =========================================================
# UI helpers
# =========================================================
def ui_result(title: str, value: str, subtitle=None):
    children = [
        html.Div(title.upper(), className="resultTitle"),
        html.Div(value, className="resultValue"),
    ]

    if subtitle:
        if isinstance(subtitle, (list, tuple)):
            for s in subtitle:
                if s is None:
                    continue
                # dash components can be appended directly
                if hasattr(s, "to_plotly_json"):
                    children.append(s)
                else:
                    children.append(html.Div(str(s), className="resultSub"))
        else:
            if hasattr(subtitle, "to_plotly_json"):
                children.append(subtitle)
            else:
                children.append(html.Div(str(subtitle), className="resultSub"))

    return html.Div(children, className="resultInner")


def get_feature_importances(model_obj):
    """Try to extract feature_importances_ from estimator or a Pipeline."""
    if model_obj is None:
        return None

    if hasattr(model_obj, "feature_importances_"):
        return getattr(model_obj, "feature_importances_", None)

    named_steps = getattr(model_obj, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in reversed(list(named_steps.values())):
            if hasattr(step, "feature_importances_"):
                return getattr(step, "feature_importances_", None)

    return None


def build_feature_importance_figure(model_obj, feature_cols: list[str], top_k: int = 12):
    imp = get_feature_importances(model_obj)
    if imp is None or feature_cols is None:
        fig = go.Figure()
        fig.update_layout(
            height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="Feature importance is not available for this model.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=13),
                )
            ],
        )
        return fig

    imp = np.array(list(imp), dtype=float)
    n = min(len(feature_cols), len(imp))
    feat = np.array(feature_cols[:n], dtype=object)
    imp = imp[:n]

    order = np.argsort(imp)[-top_k:]
    feat_top = feat[order]
    imp_top = imp[order]

    fig = go.Figure(data=[go.Bar(x=imp_top, y=feat_top, orientation="h")])
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="importance", gridcolor="rgba(148,163,184,.25)", zeroline=False),
        yaxis=dict(title="", automargin=True, gridcolor="rgba(148,163,184,.10)"),
        showlegend=False,
    )
    return fig

