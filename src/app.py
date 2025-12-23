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



# =========================================================
# Prediction range helpers
# =========================================================
def fmt_ils(v: float) -> str:
    return f"{v:,.0f} ₪"


def estimate_price_ranges(model_choice: str, price: float):
    """Return list of ranges as (label, lo, hi).

    Ranges are based on offline test metrics (MAE/RMSE) stored in MODEL_CARDS.
    They are **not** confidence intervals; just practical error-based bands.
    """
    card = MODEL_CARDS.get(model_choice, {}) if isinstance(MODEL_CARDS, dict) else {}
    mae = card.get("mae")
    rmse = card.get("rmse")

    # Taiwan predicts "price of unit area" in tutorial units
    if model_choice == "taiwan":
        delta = abs(price) * 0.10
        return [("Approx. range (±10%)", max(0.0, price - delta), price + delta)]

    # If metrics are missing (e.g., v1), fallback to ±20%
    if mae is None and rmse is None:
        delta = abs(price) * 0.20
        return [("Approx. range (±20%)", max(0.0, price - delta), price + delta)]

    ranges = []
    if mae is not None:
        d = float(mae)
        ranges.append(("Typical range (±MAE)", max(0.0, price - d), price + d))
    if rmse is not None:
        d = float(rmse)
        ranges.append(("Conservative range (±RMSE)", max(0.0, price - d), price + d))
    return ranges


def rf_quantile_ranges(model, X, quantile_pairs=((0.10, 0.90), (0.05, 0.95)), log1p_target: bool = False):
    """Estimate prediction spread from RandomForest per-tree predictions.

    Supports:
      - RandomForestRegressor (has .estimators_)
      - sklearn Pipeline ending with a RandomForestRegressor

    Returns: list of (label, low, high). Empty list if not supported.
    """
    rf = None
    Xt = X

    # Direct RandomForest
    if hasattr(model, "estimators_"):
        rf = model
        Xt = X

    # Pipeline ending with RandomForest
    elif hasattr(model, "steps") and getattr(model, "steps", None):
        try:
            last = model.steps[-1][1]
            if hasattr(last, "estimators_"):
                rf = last
                try:
                    Xt = model[:-1].transform(X)
                except Exception:
                    # If transform is unavailable for some reason, fall back to raw X
                    Xt = X
        except Exception:
            rf = None

    if rf is None:
        return []

    try:
        preds = np.array([est.predict(Xt)[0] for est in rf.estimators_], dtype=float)
        preds = preds[np.isfinite(preds)]
        if preds.size < 5:
            return []

        if log1p_target:
            preds = np.expm1(preds)

        preds = np.maximum(preds, 0.0)

        out = []
        for ql, qh in quantile_pairs:
            lo = float(np.quantile(preds, ql))
            hi = float(np.quantile(preds, qh))
            out.append((f"Model spread (P{int(round(ql*100)):02d}–P{int(round(qh*100)):02d})", lo, hi))
        return out
    except Exception:
        return []


# =========================================================
# Fallback columns
# =========================================================
TEL_AVIV_V2_FALLBACK_COLS = [
    "netArea",
    "grossArea",
    "rooms",
    "floor",
    "floors",
    "apartmentsInBuilding",
    "parking",
    "storage",
    "roof",
    "yard",
    "constructionYear",
    "tx_year",
    "tx_month",
    "tx_quarter",
    "building_age_at_tx",
    "floor_ratio",
]


# =========================================================
# Feature builder (v2/v3.2 share same schema)
# =========================================================
def build_tel_aviv_features(
    netarea, rooms, floor, year,
    gross_area=None, floors=None,
    apartments_in_building=None,
    parking=None, storage=None, roof=None, yard=None,
    feature_cols_override=None
):
    netarea = float(netarea)
    rooms = float(rooms)
    floor = float(floor)
    year = float(year)

    gross_area = float(gross_area) if gross_area not in [None, ""] else np.nan
    floors = float(floors) if floors not in [None, ""] else np.nan
    apartments_in_building = float(apartments_in_building) if apartments_in_building not in [None, ""] else np.nan
    parking = float(parking) if parking not in [None, ""] else np.nan
    storage = float(storage) if storage not in [None, ""] else np.nan
    roof = float(roof) if roof not in [None, ""] else np.nan
    yard = float(yard) if yard not in [None, ""] else np.nan

    if np.isnan(gross_area):
        gross_area = netarea

    now = datetime.now()
    tx_year = now.year
    tx_month = now.month
    tx_quarter = (tx_month - 1) // 3 + 1

    building_age_at_tx = np.nan
    if year and year > 0:
        building_age_at_tx = max(0, tx_year - int(year))

    floor_ratio = np.nan
    if not np.isnan(floors) and floors > 0:
        floor_ratio = floor / floors

    row = {
        "netArea": netarea,
        "grossArea": gross_area,
        "rooms": rooms,
        "floor": floor,
        "floors": floors,
        "apartmentsInBuilding": apartments_in_building,
        "parking": parking,
        "storage": storage,
        "roof": roof,
        "yard": yard,
        "constructionYear": year,
        "tx_year": tx_year,
        "tx_month": tx_month,
        "tx_quarter": tx_quarter,
        "building_age_at_tx": building_age_at_tx,
        "floor_ratio": floor_ratio,
    }

    feature_cols = (
        feature_cols_override
        or tel_aviv_v3_2_feature_cols
        or tel_aviv_v2_feature_cols
        or TEL_AVIV_V2_FALLBACK_COLS
    )

    X = pd.DataFrame([row])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


def validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=None):
    try:
        netarea_f = float(netarea)
        rooms_f = float(rooms)
        floor_f = float(floor)
        year_f = float(year)
    except Exception:
        return "Please enter valid numeric values for required Tel Aviv fields."

    if netarea_f <= 0 or netarea_f > 1000:
        return "Net area must be > 0 and look realistic (e.g., 20–300 m²)."
    if rooms_f <= 0 or rooms_f > 12:
        return "Rooms must be in a realistic range (e.g., 1–12)."
    if floor_f < -2 or floor_f > 200:
        return "Floor must be in a realistic range (e.g., -2 to 200)."

    current_year = datetime.now().year
    if year_f < 1900 or year_f > current_year + 1:
        return f"Construction year must be between 1900 and {current_year + 1}."

    if floors_total not in [None, ""]:
        try:
            floors_total_f = float(floors_total)
            if floors_total_f <= 0 or floors_total_f > 300:
                return "Total floors must be in a realistic range (1–300)."
            if floor_f > floors_total_f:
                return "Floor cannot be greater than total floors in building."
        except Exception:
            return "Total floors must be numeric."
    return None


