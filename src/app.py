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
APP_BUILD = "UI_v3_fixed_css_injected__2025-12-20"


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


# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Taiwan model (учтём регистр на Linux)
MODEL_TAIWAN_PATH = first_existing(
    os.path.join(MODELS_DIR, "real_estate_model.pkl"),
    os.path.join(MODELS_DIR, "Real_estate_model.pkl"),
)

# Tel Aviv v1
MODEL_TEL_AVIV_V1_PATH = os.path.join(MODELS_DIR, "tel_aviv_real_estate_model.pkl")

# Tel Aviv v2
MODEL_TEL_AVIV_V2_PATH = os.path.join(MODELS_DIR, "tel_aviv_real_estate_model_v2.pkl")
TEL_AVIV_V2_FEATS_PATH = os.path.join(MODELS_DIR, "tel_aviv_feature_cols_v2.json")

# Tel Aviv v3.2 clean (CLI artifacts)
MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_real_estate_model_v3_2_clean_cli.pkl"
)
TEL_AVIV_V3_2_FEATS_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_feature_cols_v3_2_clean_cli.json"
)
TEL_AVIV_V3_2_METRICS_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_metrics_v3_2_clean_cli.json"
)


# =========================================================
# Load models (не падаем, если чего-то нет — просто скрываем опции)
# =========================================================
model_taiwan = None
if MODEL_TAIWAN_PATH:
    try:
        model_taiwan = joblib.load(MODEL_TAIWAN_PATH)
    except Exception:
        model_taiwan = None

model_tel_aviv_v1 = None
if os.path.exists(MODEL_TEL_AVIV_V1_PATH):
    try:
        model_tel_aviv_v1 = joblib.load(MODEL_TEL_AVIV_V1_PATH)
    except Exception:
        model_tel_aviv_v1 = None

model_tel_aviv_v2 = None
tel_aviv_v2_feature_cols = None
if os.path.exists(MODEL_TEL_AVIV_V2_PATH):
    try:
        model_tel_aviv_v2 = joblib.load(MODEL_TEL_AVIV_V2_PATH)
    except Exception:
        model_tel_aviv_v2 = None
    tel_aviv_v2_feature_cols = safe_load_json(TEL_AVIV_V2_FEATS_PATH)

model_tel_aviv_v3_2 = None
tel_aviv_v3_2_feature_cols = None
tel_aviv_v3_2_metrics = None
if os.path.exists(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH):
    try:
        model_tel_aviv_v3_2 = joblib.load(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH)
    except Exception:
        model_tel_aviv_v3_2 = None
    tel_aviv_v3_2_feature_cols = safe_load_json(TEL_AVIV_V3_2_FEATS_PATH)
    tel_aviv_v3_2_metrics = safe_load_json(TEL_AVIV_V3_2_METRICS_PATH)


# =========================================================
# Metrics / Model cards (портфолио-стиль)
# =========================================================
def pick_metric(metrics: dict | None, key: str, fallback):
    if isinstance(metrics, dict) and key in metrics and metrics[key] is not None:
        return metrics[key]
    return fallback


MODEL_CARDS = {
    "tel_aviv_v3_2_clean": {
        "title": "Tel Aviv model v3.2_clean (BEST — time split + cleaning)",
        "mae": pick_metric(tel_aviv_v3_2_metrics, "mae", 928_568),
        "rmse": pick_metric(tel_aviv_v3_2_metrics, "rmse", 1_503_636),
        "r2": pick_metric(tel_aviv_v3_2_metrics, "r2", 0.575),
        "note": "RandomForest tuned + missing indicators. Time-aware split after cleaning.",
        "pill": "BEST",
    },
    "tel_aviv_v2": {
        "title": "Tel Aviv model v2 (RandomForest, log(price))",
        "mae": 669_657,
        "rmse": 1_418_990,
        "r2": 0.610,
        "note": "Engineered features + log(price).",
        "pill": "IMPROVED",
    },
    "tel_aviv_v1": {
        "title": "Tel Aviv model v1 (baseline)",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Baseline with 4 raw inputs.",
        "pill": "BASELINE",
    },
    "taiwan": {
        "title": "Taiwan tutorial model",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Classic tutorial dataset for end-to-end demo.",
        "pill": "TUTORIAL",
    },
}