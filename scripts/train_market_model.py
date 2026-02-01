import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLS = [
    "area_m2",
    "gross_area_m2",
    "rooms",
    "floors",
    "construction_year",
    "lat",
    "long",
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


def train_model(df_train: pd.DataFrame):
    X = df_train[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df_train["price"].astype(float)
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate(model, df_test: pd.DataFrame):
    X = df_test[FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True))
    y_true = df_test["price"].astype(float)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_test": int(len(df_test)),
        "trained_at": datetime.utcnow().isoformat(),
    }


def save_artifacts(model, metrics: dict, output_dir: str, model_id: str):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, f"{model_id}.pkl"))
    with open(os.path.join(output_dir, f"{model_id}_features.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, f"{model_id}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Standardized CSV (data_contract)")
    parser.add_argument("--market", required=True, choices=["us-nyc", "us-mia"], help="Market id")
    args = parser.parse_args()

    model_id = "us_nyc_v1" if args.market == "us-nyc" else "us_mia_v1"
    output_dir = os.path.join("models", "markets", args.market)

    df = load_dataset(args.input)
    train_df, test_df = time_split(df)
    model = train_model(train_df)
    metrics = evaluate(model, test_df)
    save_artifacts(model, metrics, output_dir, model_id)

    print(f"Saved {model_id} to {output_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
