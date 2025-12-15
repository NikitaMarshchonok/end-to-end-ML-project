# src/train_model.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
DEFAULT_SPLIT_DATE = "2018-03-25"

# v2/v3 feature set (16)
TEL_AVIV_V2_FEATURES = [
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


@dataclass
class CleaningReport:
    rows_before: int
    rows_after_price_filter: int
    rows_dropped_netarea_le0: int
    cy_future_fixed_to_nan: int
    gross_le0_fixed_to_nan: int
    gross_lt_net_fixed_to_nan: int


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_transaction_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # dayfirst=True критично для твоего датасета (ты это уже проверил)
    df["transactionDate"] = pd.to_datetime(df["transactionDate"], errors="coerce", dayfirst=True)
    return df


def build_tel_aviv_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # derived time features
    df["tx_year"] = df["transactionDate"].dt.year
    df["tx_month"] = df["transactionDate"].dt.month
    df["tx_quarter"] = df["transactionDate"].dt.quarter

    # building age at transaction
    df["building_age_at_tx"] = df["tx_year"] - df["constructionYear"]

    # floor ratio
    den = df["floors"].replace({0: np.nan})
    df["floor_ratio"] = df["floor"] / den

    return df


def clean_tel_aviv_df(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    df = df.copy()

    # numeric safety
    for c in ["price", "constructionYear", "netArea", "grossArea", "floor", "floors", "rooms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows_before = len(df)

    # price filter
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    df = df[(df["price"] >= 200_000) & (df["price"] <= 60_000_000)].copy()
    rows_after_price_filter = len(df)

    # fix constructionYear > tx_year -> NaN (НЕ удаляем строку)
    df["tx_year"] = df["transactionDate"].dt.year
    bad_cy_future = df["constructionYear"].notna() & df["tx_year"].notna() & (df["constructionYear"] > df["tx_year"])
    cy_future_fixed_to_nan = int(bad_cy_future.sum())
    df.loc[bad_cy_future, "constructionYear"] = np.nan

    # grossArea sanity
    bad_gross0 = df["grossArea"].notna() & (df["grossArea"] <= 0)
    gross_le0_fixed_to_nan = int(bad_gross0.sum())
    df.loc[bad_gross0, "grossArea"] = np.nan

    bad_gross_lt_net = df["grossArea"].notna() & df["netArea"].notna() & (df["grossArea"] < df["netArea"])
    gross_lt_net_fixed_to_nan = int(bad_gross_lt_net.sum())
    df.loc[bad_gross_lt_net, "grossArea"] = np.nan

    # netArea <= 0 — удаляем
    bad_net0 = df["netArea"].notna() & (df["netArea"] <= 0)
    rows_dropped_netarea_le0 = int(bad_net0.sum())
    df = df[~bad_net0].copy()

    rep = CleaningReport(
        rows_before=rows_before,
        rows_after_price_filter=rows_after_price_filter,
        rows_dropped_netarea_le0=rows_dropped_netarea_le0,
        cy_future_fixed_to_nan=cy_future_fixed_to_nan,
        gross_le0_fixed_to_nan=gross_le0_fixed_to_nan,
        gross_lt_net_fixed_to_nan=gross_lt_net_fixed_to_nan,
    )
    return df, rep


def make_v3_rf_model() -> Pipeline:
    # твои лучшие параметры v3.1
    rf = RandomForestRegressor(
        n_estimators=1200,
        max_depth=40,
        max_features=0.7,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", rf),
    ])
    return pipe


def time_split_masks(dates: pd.Series, split_date: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    train_m = dates < split_date
    test_m = dates >= split_date
    return train_m, test_m


def main():
    parser = argparse.ArgumentParser(description="Train Tel Aviv model (v3.2_clean) with time-aware evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save artifacts")
    parser.add_argument("--split-date", type=str, default=DEFAULT_SPLIT_DATE, help="YYYY-MM-DD boundary for time split")
    parser.add_argument("--out-model", type=str, default="tel_aviv_real_estate_model_v3_2_clean.pkl")
    parser.add_argument("--out-metrics", type=str, default="tel_aviv_metrics_v3_2_clean.json")
    parser.add_argument("--out-feats", type=str, default="tel_aviv_feature_cols_v3_2_clean.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    split_date = pd.Timestamp(args.split_date)

    print("Loading:", dataset_path)
    df = pd.read_csv(dataset_path)

    df = parse_transaction_date(df)
    df = df[df["transactionDate"].notna()].copy()

    df, rep = clean_tel_aviv_df(df)
    df = build_tel_aviv_v2_features(df)

    # safety: remove inf
    for c in TEL_AVIV_V2_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TEL_AVIV_V2_FEATURES] = df[TEL_AVIV_V2_FEATURES].replace([np.inf, -np.inf], np.nan)

    X = df[TEL_AVIV_V2_FEATURES]
    y_raw = df["price"].values
    y = np.log1p(y_raw)

    train_m, test_m = time_split_masks(df["transactionDate"], split_date)

    Xtr, Xte = X.loc[train_m], X.loc[test_m]
    ytr, yte = y[train_m.values], y[test_m.values]
    yte_raw = y_raw[test_m.values]

    print(f"Split date: {split_date} | Train: {len(Xtr)} | Test: {len(Xte)}")

    model = make_v3_rf_model()
    model.fit(Xtr, ytr)
    pred = np.expm1(model.predict(Xte))

    mae = mean_absolute_error(yte_raw, pred)
    rmse = float(np.sqrt(mean_squared_error(yte_raw, pred)))
    r2 = r2_score(yte_raw, pred)

    print("\n✅ v3.2_clean time-split metrics")
    print(f"MAE : {mae:,.0f} NIS")
    print(f"RMSE: {rmse:,.0f} NIS")
    print(f"R2  : {r2:.4f}")

    # train on FULL cleaned data for final artifact
    model_full = make_v3_rf_model()
    model_full.fit(X, y)

    out_model = models_dir / args.out_model
    joblib.dump(model_full, out_model)

    out_feats = models_dir / args.out_feats
    out_feats.write_text(json.dumps(TEL_AVIV_V2_FEATURES, indent=2), encoding="utf-8")

    metrics = {
        "version": "v3.2_clean",
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "split_date": str(split_date),
        "rows_before": rep.rows_before,
        "rows_after_price_filter": rep.rows_after_price_filter,
        "rows_dropped_netarea_le0": rep.rows_dropped_netarea_le0,
        "constructionYear_future_fixed_to_nan": rep.cy_future_fixed_to_nan,
        "grossArea_le0_fixed_to_nan": rep.gross_le0_fixed_to_nan,
        "grossArea_lt_netArea_fixed_to_nan": rep.gross_lt_net_fixed_to_nan,
        "train_size": int(train_m.sum()),
        "test_size": int(test_m.sum()),
        "test_mae_nis": float(mae),
        "test_rmse_nis": float(rmse),
        "test_r2": float(r2),
        "features": TEL_AVIV_V2_FEATURES,
        "note": "RandomForest tuned + SimpleImputer(add_indicator=True), trained on log1p(price).",
    }

    out_metrics = models_dir / args.out_metrics
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n✅ Saved model:", out_model)
    print("✅ Saved metrics:", out_metrics)
    print("✅ Saved feature cols:", out_feats)


if __name__ == "__main__":
    main()
