import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def build_tel_aviv_row(feature_cols):
    # фиксируем "текущую дату" для стабильного теста
    tx_year, tx_month = 2024, 1
    tx_quarter = 1

    # минимально реалистичный объект
    netArea = 85.0
    grossArea = 95.0
    rooms = 3.0
    floor = 5.0
    floors = 12.0
    apartmentsInBuilding = 40.0
    parking = 1.0
    storage = 1.0
    roof = np.nan
    yard = np.nan
    constructionYear = 2008.0

    building_age_at_tx = max(0, int(tx_year - constructionYear))
    floor_ratio = floor / floors if floors and floors > 0 else np.nan

    row = {
        "netArea": netArea,
        "grossArea": grossArea,
        "rooms": rooms,
        "floor": floor,
        "floors": floors,
        "apartmentsInBuilding": apartmentsInBuilding,
        "parking": parking,
        "storage": storage,
        "roof": roof,
        "yard": yard,
        "constructionYear": constructionYear,
        "tx_year": tx_year,
        "tx_month": tx_month,
        "tx_quarter": tx_quarter,
        "building_age_at_tx": building_age_at_tx,
        "floor_ratio": floor_ratio,
    }

    X = pd.DataFrame([row])

    # гарантируем нужные колонки и порядок
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    return X

def test_v3_2_clean_predicts_positive_price():
    model_path = ROOT / "models" / "tel_aviv_real_estate_model_v3_2_clean_cli.pkl"
    feats_path = ROOT / "models" / "tel_aviv_feature_cols_v3_2_clean_cli.json"

    model = joblib.load(model_path)
    feature_cols = json.loads(feats_path.read_text())

    X = build_tel_aviv_row(feature_cols)
    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))

    assert np.isfinite(pred_price)
    assert pred_price > 0
