import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    version: str
    currency: str
    model: Any
    feature_cols: list[str] | None
    log1p_target: bool
    metrics: dict | None


@dataclass(frozen=True)
class PredictionResult:
    price: float
    p10: float
    p50: float
    p90: float
    currency: str
    model_version: str
    factors: list[dict]
    ranges: list[tuple[str, float, float]]


# =========================================================
# Paths
# =========================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")

TEL_AVIV_DATA_PATH = os.path.join(DATA_DIR, "Real_estate_Tel_Aviv_20_years.csv")
TAIWAN_DATA_PATH = os.path.join(DATA_DIR, "Real_Estate.csv")


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


# Taiwan model (case insensitive on Linux)
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
# Feature schemas (for API / UI)
# =========================================================
TEL_AVIV_REQUIRED_FEATURES = [
    {
        "name": "netArea",
        "type": "number",
        "label": "Net area",
        "min": 10,
        "max": 1000,
        "step": 1,
        "unit": "m²",
        "placeholder": "80",
        "required": True,
    },
    {
        "name": "rooms",
        "type": "number",
        "label": "Rooms",
        "min": 1,
        "max": 12,
        "step": 0.5,
        "placeholder": "3",
        "required": True,
    },
    {
        "name": "floor",
        "type": "number",
        "label": "Floor",
        "min": -2,
        "max": 200,
        "step": 1,
        "placeholder": "4",
        "required": True,
    },
    {
        "name": "constructionYear",
        "type": "number",
        "label": "Construction year",
        "min": 1900,
        "max": 2100,
        "step": 1,
        "placeholder": "2010",
        "required": True,
    },
]

TEL_AVIV_OPTIONAL_FEATURES = [
    {
        "name": "grossArea",
        "type": "number",
        "label": "Gross area",
        "min": 10,
        "max": 1200,
        "step": 1,
        "unit": "m²",
        "placeholder": "95",
        "required": False,
    },
    {
        "name": "floors",
        "type": "number",
        "label": "Total floors",
        "min": 1,
        "max": 300,
        "step": 1,
        "placeholder": "12",
        "required": False,
    },
    {
        "name": "apartmentsInBuilding",
        "type": "number",
        "label": "Apartments in building",
        "min": 1,
        "max": 500,
        "step": 1,
        "placeholder": "40",
        "required": False,
    },
    {
        "name": "parking",
        "type": "number",
        "label": "Parking spots",
        "min": 0,
        "max": 5,
        "step": 1,
        "placeholder": "1",
        "required": False,
    },
    {
        "name": "storage",
        "type": "number",
        "label": "Storage",
        "min": 0,
        "max": 3,
        "step": 1,
        "placeholder": "1",
        "required": False,
    },
    {
        "name": "roof",
        "type": "number",
        "label": "Roof area",
        "min": 0,
        "max": 300,
        "step": 1,
        "unit": "m²",
        "placeholder": "20",
        "required": False,
    },
    {
        "name": "yard",
        "type": "number",
        "label": "Yard area",
        "min": 0,
        "max": 300,
        "step": 1,
        "unit": "m²",
        "placeholder": "30",
        "required": False,
    },
]

TAIWAN_FEATURES = [
    {
        "name": "distance",
        "type": "number",
        "label": "Distance to MRT",
        "min": 0,
        "max": 20000,
        "step": 50,
        "unit": "m",
        "placeholder": "400",
        "required": True,
    },
    {
        "name": "convenience",
        "type": "number",
        "label": "Convenience stores",
        "min": 0,
        "max": 20,
        "step": 1,
        "placeholder": "4",
        "required": True,
    },
    {
        "name": "lat",
        "type": "number",
        "label": "Latitude",
        "min": 20,
        "max": 30,
        "step": 0.0001,
        "placeholder": "24.98",
        "required": True,
    },
    {
        "name": "long",
        "type": "number",
        "label": "Longitude",
        "min": 120,
        "max": 130,
        "step": 0.0001,
        "placeholder": "121.54",
        "required": True,
    },
]

# =========================================================
# Comparable sales display fields
# =========================================================
TEL_AVIV_COMPARABLE_FIELDS = [
    {"key": "netArea", "label": "Net area", "unit": "m²"},
    {"key": "rooms", "label": "Rooms"},
    {"key": "floor", "label": "Floor"},
    {"key": "constructionYear", "label": "Year"},
    {"key": "grossArea", "label": "Gross area", "unit": "m²"},
    {"key": "floors", "label": "Total floors"},
    {"key": "parking", "label": "Parking"},
]

TAIWAN_COMPARABLE_FIELDS = [
    {"key": "distance", "label": "Distance to MRT", "unit": "m"},
    {"key": "convenience", "label": "Convenience stores"},
    {"key": "lat", "label": "Latitude"},
    {"key": "long", "label": "Longitude"},
]

TEL_AVIV_DRIFT_FEATURES = ["netArea", "rooms", "floor", "constructionYear"]
TAIWAN_DRIFT_FEATURES = ["distance", "convenience", "lat", "long"]


# =========================================================
# Load models
# =========================================================
def _safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None


def _load_models() -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {}

    if MODEL_TAIWAN_PATH:
        model_taiwan = _safe_joblib_load(MODEL_TAIWAN_PATH)
        if model_taiwan is not None:
            models["taiwan"] = ModelSpec(
                id="taiwan",
                name="Taiwan tutorial model",
                version="1.0",
                currency="TWD",
                model=model_taiwan,
                feature_cols=None,
                log1p_target=False,
                metrics=None,
            )

    if os.path.exists(MODEL_TEL_AVIV_V1_PATH):
        model_v1 = _safe_joblib_load(MODEL_TEL_AVIV_V1_PATH)
        if model_v1 is not None:
            models["tel_aviv_v1"] = ModelSpec(
                id="tel_aviv_v1",
                name="Tel Aviv model v1 (baseline)",
                version="1.0",
                currency="ILS",
                model=model_v1,
                feature_cols=None,
                log1p_target=False,
                metrics=None,
            )

    if os.path.exists(MODEL_TEL_AVIV_V2_PATH):
        model_v2 = _safe_joblib_load(MODEL_TEL_AVIV_V2_PATH)
        if model_v2 is not None:
            models["tel_aviv_v2"] = ModelSpec(
                id="tel_aviv_v2",
                name="Tel Aviv model v2 (log price)",
                version="2.0",
                currency="ILS",
                model=model_v2,
                feature_cols=safe_load_json(TEL_AVIV_V2_FEATS_PATH),
                log1p_target=True,
                metrics=None,
            )

    if os.path.exists(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH):
        model_v3 = _safe_joblib_load(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH)
        if model_v3 is not None:
            models["tel_aviv_v3_2_clean"] = ModelSpec(
                id="tel_aviv_v3_2_clean",
                name="Tel Aviv model v3.2_clean (best)",
                version="3.2",
                currency="ILS",
                model=model_v3,
                feature_cols=safe_load_json(TEL_AVIV_V3_2_FEATS_PATH),
                log1p_target=True,
                metrics=safe_load_json(TEL_AVIV_V3_2_METRICS_PATH),
            )

    return models


MODEL_REGISTRY = _load_models()


# =========================================================
# Feature engineering
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
        or (MODEL_REGISTRY.get("tel_aviv_v3_2_clean") or ModelSpec("", "", "", "", None, None, False, None)).feature_cols
        or (MODEL_REGISTRY.get("tel_aviv_v2") or ModelSpec("", "", "", "", None, None, False, None)).feature_cols
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


# =========================================================
# Quantiles / ranges
# =========================================================
def estimate_price_ranges(metrics: dict | None, price: float, model_id: str):
    if model_id == "taiwan":
        delta = abs(price) * 0.10
        return [("Approx. range (±10%)", max(0.0, price - delta), price + delta)]

    mae = metrics.get("mae") if isinstance(metrics, dict) else None
    rmse = metrics.get("rmse") if isinstance(metrics, dict) else None

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


def rf_quantiles(model, X, quantiles=(0.10, 0.50, 0.90), log1p_target: bool = False):
    rf = None
    Xt = X

    if hasattr(model, "estimators_"):
        rf = model
        Xt = X
    elif hasattr(model, "steps") and getattr(model, "steps", None):
        try:
            last = model.steps[-1][1]
            if hasattr(last, "estimators_"):
                rf = last
                try:
                    Xt = model[:-1].transform(X)
                except Exception:
                    Xt = X
        except Exception:
            rf = None

    if rf is None:
        return None

    try:
        preds = np.array([est.predict(Xt)[0] for est in rf.estimators_], dtype=float)
        preds = preds[np.isfinite(preds)]
        if preds.size < 5:
            return None

        if log1p_target:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0.0)

        return [float(np.quantile(preds, q)) for q in quantiles]
    except Exception:
        return None


# =========================================================
# Explainability (lightweight)
# =========================================================
def get_feature_importances(model_obj):
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


def build_local_factors(model_id: str, base_features: dict, base_price: float):
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.model is None:
        return []

    if model_id in ("tel_aviv_v2", "tel_aviv_v3_2_clean"):
        feature_cols = spec.feature_cols or TEL_AVIV_V2_FALLBACK_COLS
        importance = get_feature_importances(spec.model)
        if importance is None:
            candidates = [f["name"] for f in TEL_AVIV_REQUIRED_FEATURES]
        else:
            imp = np.array(list(importance), dtype=float)
            n = min(len(feature_cols), len(imp))
            order = np.argsort(imp[:n])[-6:]
            candidates = [feature_cols[i] for i in order if i < len(feature_cols)]
    elif model_id == "tel_aviv_v1":
        candidates = ["netArea", "rooms", "floor", "constructionYear"]
    else:
        candidates = ["distance", "convenience", "lat", "long"]

    def _get_step(name: str):
        all_features = TEL_AVIV_REQUIRED_FEATURES + TEL_AVIV_OPTIONAL_FEATURES + TAIWAN_FEATURES
        for f in all_features:
            if f["name"] == name:
                return f.get("step") or 1
        return 1

    factors = []
    for name in candidates[:5]:
        val = base_features.get(name)
        if val in [None, ""]:
            continue

        step = _get_step(name)
        try:
            step = float(step)
        except Exception:
            step = 1

        bumped = dict(base_features)
        try:
            bumped[name] = float(val) + step
        except Exception:
            continue

        try:
            bumped_price = _predict_price_raw(model_id, bumped)
        except Exception:
            continue

        diff = bumped_price - base_price
        direction = "up" if diff >= 0 else "down"
        impact_pct = 0 if base_price <= 0 else min(100, abs(diff) / base_price * 100)
        factors.append(
            {
                "name": name,
                "impact": int(round(impact_pct)),
                "direction": direction,
            }
        )

    return factors


# =========================================================
# Public API
# =========================================================
def list_models():
    out = []
    for spec in MODEL_REGISTRY.values():
        if spec.model is None:
            continue

        if spec.id in ("tel_aviv_v2", "tel_aviv_v3_2_clean"):
            features = TEL_AVIV_REQUIRED_FEATURES + TEL_AVIV_OPTIONAL_FEATURES
        elif spec.id == "tel_aviv_v1":
            features = TEL_AVIV_REQUIRED_FEATURES
        else:
            features = TAIWAN_FEATURES

        out.append(
            {
                "id": spec.id,
                "name": spec.name,
                "version": spec.version,
                "currency": spec.currency,
                "features": features,
            }
        )
    return out


def get_model_spec(model_id: str) -> ModelSpec | None:
    return MODEL_REGISTRY.get(model_id)


def _predict_price_raw(model_id: str, features: dict) -> float:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.model is None:
        raise ValueError("Model not available.")

    if model_id in ("tel_aviv_v2", "tel_aviv_v3_2_clean"):
        err = validate_tel_aviv_inputs(
            features.get("netArea"),
            features.get("rooms"),
            features.get("floor"),
            features.get("constructionYear"),
            floors_total=features.get("floors"),
        )
        if err:
            raise ValueError(err)

        X = build_tel_aviv_features(
            features.get("netArea"),
            features.get("rooms"),
            features.get("floor"),
            features.get("constructionYear"),
            gross_area=features.get("grossArea"),
            floors=features.get("floors"),
            apartments_in_building=features.get("apartmentsInBuilding"),
            parking=features.get("parking"),
            storage=features.get("storage"),
            roof=features.get("roof"),
            yard=features.get("yard"),
            feature_cols_override=spec.feature_cols,
        )
        pred = float(spec.model.predict(X)[0])
        price = float(np.expm1(pred)) if spec.log1p_target else pred
        return max(price, 0.0)

    if model_id == "tel_aviv_v1":
        err = validate_tel_aviv_inputs(
            features.get("netArea"),
            features.get("rooms"),
            features.get("floor"),
            features.get("constructionYear"),
        )
        if err:
            raise ValueError(err)
        X = np.array(
            [[
                float(features.get("netArea")),
                float(features.get("rooms")),
                float(features.get("floor")),
                float(features.get("constructionYear")),
            ]]
        )
        price = float(spec.model.predict(X)[0])
        return max(price, 0.0)

    if model_id == "taiwan":
        missing = [k for k in ["distance", "convenience", "lat", "long"] if features.get(k) in [None, ""]]
        if missing:
            raise ValueError("Please fill all Taiwan fields.")
        X = np.array(
            [[
                float(features.get("distance")),
                float(features.get("convenience")),
                float(features.get("lat")),
                float(features.get("long")),
            ]]
        )
        pred = float(spec.model.predict(X)[0])
        return max(pred, 0.0)

    raise ValueError("Unsupported model.")


def predict(model_id: str, features: dict) -> PredictionResult:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.model is None:
        raise ValueError("Model not available.")

    price = _predict_price_raw(model_id, features)
    p10 = p50 = p90 = price

    if model_id in ("tel_aviv_v2", "tel_aviv_v3_2_clean"):
        X = build_tel_aviv_features(
            features.get("netArea"),
            features.get("rooms"),
            features.get("floor"),
            features.get("constructionYear"),
            gross_area=features.get("grossArea"),
            floors=features.get("floors"),
            apartments_in_building=features.get("apartmentsInBuilding"),
            parking=features.get("parking"),
            storage=features.get("storage"),
            roof=features.get("roof"),
            yard=features.get("yard"),
            feature_cols_override=spec.feature_cols,
        )
        quantiles = rf_quantiles(spec.model, X, log1p_target=True)
        if quantiles:
            p10, p50, p90 = quantiles
        else:
            ranges = estimate_price_ranges(spec.metrics, price, model_id)
            if ranges:
                p10 = ranges[0][1]
                p90 = ranges[0][2]
                p50 = price

    elif model_id == "tel_aviv_v1":
        ranges = estimate_price_ranges(spec.metrics, price, model_id)
        if ranges:
            p10 = ranges[0][1]
            p90 = ranges[0][2]
            p50 = price

    else:
        ranges = estimate_price_ranges(spec.metrics, price, model_id)
        if ranges:
            p10 = ranges[0][1]
            p90 = ranges[0][2]
            p50 = price

    ranges = estimate_price_ranges(spec.metrics, price, model_id)
    factors = build_local_factors(model_id, features, price)

    return PredictionResult(
        price=price,
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        currency=spec.currency,
        model_version=spec.version,
        factors=factors,
        ranges=ranges,
    )


# =========================================================
# Comparable sales
# =========================================================
def _prepare_comparables_index(df: pd.DataFrame, feature_cols: list[str]):
    df_numeric = df.copy()
    for col in feature_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

    medians = df_numeric[feature_cols].median(numeric_only=True)
    X = df_numeric[feature_cols].fillna(medians)
    means = X.mean(numeric_only=True)
    stds = X.std(numeric_only=True).replace(0, 1)

    return {
        "df": df_numeric,
        "X": X,
        "means": means,
        "stds": stds,
        "feature_cols": feature_cols,
    }


def _load_tel_aviv_comparables():
    if not os.path.exists(TEL_AVIV_DATA_PATH):
        return None

    df = pd.read_csv(TEL_AVIV_DATA_PATH)
    cols = [
        "price",
        "netArea",
        "rooms",
        "floor",
        "constructionYear",
        "grossArea",
        "floors",
        "apartmentsInBuilding",
        "parking",
        "storage",
        "roof",
        "yard",
    ]
    df = df[cols].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)]
    return _prepare_comparables_index(df, [c for c in cols if c != "price"])


def _load_taiwan_comparables():
    if not os.path.exists(TAIWAN_DATA_PATH):
        return None

    df = pd.read_csv(TAIWAN_DATA_PATH)
    rename = {
        "Distance to the nearest MRT station": "distance",
        "Number of convenience stores": "convenience",
        "Latitude": "lat",
        "Longitude": "long",
        "House price of unit area": "price",
    }
    df = df.rename(columns=rename)
    cols = ["price", "distance", "convenience", "lat", "long"]
    df = df[cols].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)]
    return _prepare_comparables_index(df, [c for c in cols if c != "price"])


TEL_AVIV_COMPARABLES_INDEX = _load_tel_aviv_comparables()
TAIWAN_COMPARABLES_INDEX = _load_taiwan_comparables()


def get_comparables(model_id: str, features: dict, top_k: int = 5):
    if model_id in ("tel_aviv_v1", "tel_aviv_v2", "tel_aviv_v3_2_clean"):
        index = TEL_AVIV_COMPARABLES_INDEX
        display_fields = TEL_AVIV_COMPARABLE_FIELDS
        currency = "ILS"
    elif model_id == "taiwan":
        index = TAIWAN_COMPARABLES_INDEX
        display_fields = TAIWAN_COMPARABLE_FIELDS
        currency = "TWD"
    else:
        return {"currency": "ILS", "fields": [], "items": []}

    if index is None:
        return {"currency": currency, "fields": display_fields, "items": []}

    feature_cols = index["feature_cols"]
    available = [f for f in feature_cols if features.get(f) not in [None, ""]]
    if len(available) < 2:
        return {"currency": currency, "fields": display_fields, "items": []}

    try:
        input_vals = [float(features.get(f)) for f in available]
    except Exception:
        return {"currency": currency, "fields": display_fields, "items": []}

    X = index["X"][available].to_numpy()
    means = index["means"][available].to_numpy()
    stds = index["stds"][available].to_numpy()

    Xz = (X - means) / stds
    vz = (np.array(input_vals, dtype=float) - means) / stds
    dists = np.linalg.norm(Xz - vz, axis=1)

    k = max(1, min(int(top_k), 10))
    idx = np.argsort(dists)[:k]
    def _clean_value(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and np.isnan(v):
                return None
        except Exception:
            pass
        return v

    items = []
    for pos in idx:
        row = index["df"].iloc[pos]
        item_features = {f: _clean_value(row.get(f)) for f in feature_cols}
        items.append(
            {
                "price": float(row.get("price", 0)),
                "distance": float(dists[pos]),
                "features": item_features,
            }
        )

    return {"currency": currency, "fields": display_fields, "items": items}


# =========================================================
# Monitoring / drift
# =========================================================
def get_baseline_stats(model_id: str):
    if model_id in ("tel_aviv_v1", "tel_aviv_v2", "tel_aviv_v3_2_clean"):
        if not os.path.exists(TEL_AVIV_DATA_PATH):
            return None
        cols = TEL_AVIV_DRIFT_FEATURES
        df = pd.read_csv(TEL_AVIV_DATA_PATH, usecols=[c for c in cols if c])
        stats = {}
        for col in cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std() or 1.0),
            }
        return stats

    if model_id == "taiwan":
        if not os.path.exists(TAIWAN_DATA_PATH):
            return None
        df = pd.read_csv(TAIWAN_DATA_PATH)
        rename = {
            "Distance to the nearest MRT station": "distance",
            "Number of convenience stores": "convenience",
            "Latitude": "lat",
            "Longitude": "long",
        }
        df = df.rename(columns=rename)
        stats = {}
        for col in TAIWAN_DRIFT_FEATURES:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std() or 1.0),
            }
        return stats

    return None
