import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_NUMERIC_COLS = [
    "area_m2",
    "gross_area_m2",
    "rooms",
    "floors",
    "construction_year",
    "lat",
    "long",
]

ENGINEERED_NUMERIC_COLS = [
    "area_log",
    "gross_to_net",
    "age_at_tx",
    "tx_year",
    "tx_month",
    "tx_quarter",
]

NUMERIC_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

CATEGORICAL_COLS = [
    "neighborhood",
    "building_class",
    "property_type",
]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["price"].notna()]
    df = df[df["area_m2"].notna()]
    return df


def time_split(df: pd.DataFrame, date_col: str = "transaction_date", test_size: float = 0.2):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gross_area_m2"] = out["gross_area_m2"].fillna(out["area_m2"])
    out["area_log"] = np.log1p(out["area_m2"].clip(lower=0))
    out["gross_to_net"] = out["gross_area_m2"] / out["area_m2"].replace(0, np.nan)
    out["gross_to_net"] = out["gross_to_net"].replace([np.inf, -np.inf], np.nan)
    out["gross_to_net"] = out["gross_to_net"].fillna(1.0).clip(lower=0.5, upper=5.0)

    tx_dt = pd.to_datetime(out["transaction_date"], errors="coerce")
    out["tx_year"] = tx_dt.dt.year
    out["tx_month"] = tx_dt.dt.month
    out["tx_quarter"] = tx_dt.dt.quarter
    out["age_at_tx"] = out["tx_year"] - out["construction_year"]
    out["age_at_tx"] = out["age_at_tx"].clip(lower=0)
    return out


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    X_num = df[NUMERIC_COLS].copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    X_cat = df[CATEGORICAL_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_cat[col] = X_cat[col].fillna("Unknown").astype(str)

    # Fit encoder on train categories only
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", encoder, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    preprocessor.fit(pd.concat([X_num, X_cat], axis=1))
    return preprocessor


def train_model(df_train: pd.DataFrame, log_target: bool, version: str):
    df_train = enrich_features(df_train)
    X_num = df_train[NUMERIC_COLS].copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_cat = df_train[CATEGORICAL_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_cat[col] = X_cat[col].fillna("Unknown").astype(str)
    X = pd.concat([X_num, X_cat], axis=1)
    preprocessor = build_preprocessor(df_train)
    y = df_train["price"].astype(float)
    if log_target:
        y = np.log1p(y)

    if version == "v3":
        model = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return pipeline, feature_cols


def evaluate(model, df_test: pd.DataFrame, log_target: bool):
    df_test = enrich_features(df_test)
    X_num = df_test[NUMERIC_COLS].copy()
    X_num = X_num.fillna(X_num.median(numeric_only=True))
    X_cat = df_test[CATEGORICAL_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_cat[col] = X_cat[col].fillna("Unknown").astype(str)
    X = pd.concat([X_num, X_cat], axis=1)
    y_true = df_test["price"].astype(float)
    y_pred = model.predict(X)
    if log_target:
        y_pred = np.expm1(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_test": int(len(df_test)),
        "trained_at": datetime.now().astimezone().isoformat(),
    }


def save_artifacts(model, metrics: dict, feature_cols: list[str], output_dir: str, model_id: str):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, f"{model_id}.pkl"))
    with open(os.path.join(output_dir, f"{model_id}_features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"{model_id}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Standardized CSV (data_contract)")
    parser.add_argument("--market", required=True, choices=["us-nyc", "us-mia"], help="Market id")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"], help="Model version")
    parser.add_argument("--no-log-target", action="store_true", help="Disable log1p target transform")
    args = parser.parse_args()

    model_id = f"{'us_nyc' if args.market == 'us-nyc' else 'us_mia'}_{args.version}"
    output_dir = os.path.join("models", "markets", args.market)

    df = load_dataset(args.input)
    df = df[df["price"] > 100_000]
    df = df[df["area_m2"].between(10, 2000)]
    df = df[df["transaction_date"].notna()]
    if "rooms" in df.columns:
        df = df[(df["rooms"].isna()) | (df["rooms"].between(0, 15))]
    if "floors" in df.columns:
        df = df[(df["floors"].isna()) | (df["floors"].between(1, 120))]
    upper = df["price"].quantile(0.99)
    df = df[df["price"] <= upper]
    if len(df) < 500:
        raise ValueError("Dataset too small after filtering. Check ETL mapping/filters.")
    train_df, test_df = time_split(df)
    log_target = not args.no_log_target
    model, feature_cols = train_model(train_df, log_target=log_target, version=args.version)
    metrics = evaluate(model, test_df, log_target=log_target)
    metrics["log_target"] = log_target
    metrics["price_filter_min"] = 100_000
    metrics["price_filter_p99"] = float(upper)
    save_artifacts(model, metrics, feature_cols, output_dir, model_id)

    print(f"Saved {model_id} to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
