import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def apply_training_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["price"] > 100_000]
    out = out[out["area_m2"].between(10, 2000)]
    out = out[out["transaction_date"].notna()]
    if "rooms" in out.columns:
        out = out[(out["rooms"].isna()) | (out["rooms"].between(0, 15))]
    if "floors" in out.columns:
        out = out[(out["floors"].isna()) | (out["floors"].between(1, 120))]
    upper = out["price"].quantile(0.99)
    out = out[out["price"] <= upper]
    return out


def time_split(df: pd.DataFrame, date_col: str = "transaction_date", test_size: float = 0.2):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].sort_values(date_col)
    split_idx = int(len(out) * (1 - test_size))
    return out.iloc[:split_idx], out.iloc[split_idx:]


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

    for col in CATEGORICAL_COLS:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = out[col].fillna("Unknown").astype(str)
    return out


def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def evaluate_slice(y_true: pd.Series, y_pred: pd.Series) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "mape": None, "r2": None}
    y_true_arr = y_true.astype(float).values
    y_pred_arr = y_pred.astype(float).values
    denom = np.where(y_true_arr > 0, y_true_arr, np.nan)
    ape = np.abs(y_true_arr - y_pred_arr) / denom
    return {
        "n": int(len(y_true_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr) ** 0.5),
        "mape": float(np.nanmean(ape)) if np.isfinite(np.nanmean(ape)) else None,
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else None,
    }


def evaluate_model(model_path: str, metrics_path: str, test_df: pd.DataFrame) -> dict:
    model_id = os.path.basename(model_path).replace(".pkl", "")
    model = joblib.load(model_path)
    meta = safe_load_json(metrics_path) or {}
    log_target = bool(meta.get("log_target", False))

    X = enrich_features(test_df)
    y_true = test_df["price"].astype(float)
    y_pred = pd.Series(model.predict(X), index=test_df.index)
    if log_target:
        y_pred = np.expm1(y_pred)
    y_pred = pd.Series(np.maximum(y_pred, 0), index=test_df.index)

    overall = evaluate_slice(y_true, y_pred)

    price_bucket = pd.qcut(y_true, q=5, duplicates="drop")
    by_price = []
    for bucket in sorted(price_bucket.dropna().unique()):
        mask = price_bucket == bucket
        row = evaluate_slice(y_true[mask], y_pred[mask])
        row["bucket"] = str(bucket)
        by_price.append(row)

    area_bins = [0, 50, 80, 120, 200, 400, float("inf")]
    area_labels = ["0-50", "50-80", "80-120", "120-200", "200-400", "400+"]
    area_bucket = pd.cut(test_df["area_m2"], bins=area_bins, labels=area_labels, include_lowest=True)
    by_area = []
    for label in area_labels:
        mask = area_bucket.astype(str) == label
        row = evaluate_slice(y_true[mask], y_pred[mask])
        row["bucket"] = label
        by_area.append(row)

    by_neighborhood = []
    if "neighborhood" in test_df.columns:
        top_neighborhoods = (
            test_df["neighborhood"].fillna("Unknown").astype(str).value_counts().head(10).index.tolist()
        )
        for nbh in top_neighborhoods:
            mask = test_df["neighborhood"].fillna("Unknown").astype(str) == nbh
            row = evaluate_slice(y_true[mask], y_pred[mask])
            row["neighborhood"] = nbh
            by_neighborhood.append(row)
        by_neighborhood.sort(
            key=lambda row: (row["mae"] is None, row["mae"] if row["mae"] is not None else float("inf"))
        )

    return {
        "model_id": model_id,
        "log_target": log_target,
        "overall": overall,
        "by_price_bucket": by_price,
        "by_area_bucket": by_area,
        "by_neighborhood_top10": by_neighborhood,
    }


def discover_models(market: str, model_ids: list[str] | None):
    market_dir = os.path.join("models", "markets", market)
    if model_ids:
        return [os.path.join(market_dir, f"{model_id}.pkl") for model_id in model_ids]
    if not os.path.isdir(market_dir):
        return []
    return sorted(
        [
            os.path.join(market_dir, file_name)
            for file_name in os.listdir(market_dir)
            if file_name.endswith(".pkl")
        ]
    )


def print_summary(results: list[dict]):
    print("\nModel comparison (overall):")
    print("model_id | MAE | RMSE | MAPE | R2 | n")
    print("---|---:|---:|---:|---:|---:")
    for item in sorted(results, key=lambda x: x["overall"]["mae"]):
        o = item["overall"]
        print(
            f"{item['model_id']} | {o['mae']:.2f} | {o['rmse']:.2f} | "
            f"{(o['mape'] * 100 if o['mape'] is not None else float('nan')):.2f}% | {o['r2']:.4f} | {o['n']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Standardized CSV")
    parser.add_argument("--market", required=True, choices=["us-nyc", "us-mia"], help="Market id")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids to evaluate. Default: all in models/markets/<market>/",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Time-split holdout size")
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default: reports/eval_<market>_<timestamp>.json",
    )
    args = parser.parse_args()

    df = load_dataset(args.input)
    df = apply_training_filters(df)
    if len(df) < 500:
        raise ValueError("Dataset too small after filtering.")
    _, test_df = time_split(df, test_size=args.test_size)
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()] or None
    model_paths = discover_models(args.market, model_ids)
    if not model_paths:
        raise ValueError("No model artifacts found for evaluation.")

    results = []
    missing = []
    for model_path in model_paths:
        metrics_path = model_path.replace(".pkl", "_metrics.json")
        if not os.path.exists(model_path):
            missing.append(model_path)
            continue
        results.append(evaluate_model(model_path, metrics_path, test_df))

    if not results:
        raise ValueError("No models were evaluated successfully.")

    print_summary(results)

    payload = {
        "market": args.market,
        "input": args.input,
        "test_size": args.test_size,
        "n_total_after_filters": int(len(df)),
        "n_test": int(len(test_df)),
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "models": results,
        "missing_models": missing,
    }

    out_path = args.out.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("reports", f"eval_{args.market}_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation report: {out_path}")


if __name__ == "__main__":
    main()
