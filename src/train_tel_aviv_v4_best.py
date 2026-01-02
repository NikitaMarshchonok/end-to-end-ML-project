import os
import json
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


# =========================================================
# Paths / Config
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Если у тебя датасет в другом месте — укажи через env TEL_AVIV_DATASET при запуске
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "tel_aviv.csv")

OUT_MODEL_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_real_estate_model_v4_best.pkl")
OUT_METRICS_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_v4_best_metrics.json")
OUT_FEATURES_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_v4_best_feature_cols.json")

# ✅ Если твоя колонка цены называется иначе — добавь сюда
TARGET_CANDIDATES = ["price", "Price", "deal_price", "transaction_price", "final_price", "price_nis", "price_ils"]

# ✅ Если у тебя есть колонка даты — добавь/проверь названия здесь
DATE_CANDIDATES = ["date", "deal_date", "transaction_date", "sale_date", "created_at", "timestamp"]


@dataclass
class EvalResult:
    name: str
    mae: float
    rmse: float


def _find_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _detect_target(df: pd.DataFrame) -> str:
    col = _find_first_existing(df.columns.tolist(), TARGET_CANDIDATES)
    if col is None:
        raise ValueError(
            f"Не нашёл target-колонку (цену). Ожидал одну из: {TARGET_CANDIDATES}. "
            f"Первые колонки датасета: {list(df.columns)[:40]}"
        )
    return col


def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    col = _find_first_existing(df.columns.tolist(), DATE_CANDIDATES)
    if col is None:
        return None

    parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
    if parsed.notna().mean() < 0.5:
        return None

    df[col] = parsed
    return col


def _make_onehot() -> OneHotEncoder:
    """
    Совместимость со старыми/новыми sklearn:
    - новые: sparse_output
    - старые: sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot()),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def _train_test_time_split(df: pd.DataFrame, date_col: Optional[str], test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if date_col is not None:
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _metrics_on_price(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Tuple[float, float]:
    """
    Считаем метрики в реальных деньгах:
    y хранится в log1p(price)
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def evaluate_candidate(name: str, model, X_train, y_train, X_test, y_test) -> Tuple[EvalResult, Any]:
    pipe = Pipeline(steps=[
        ("pre", _make_preprocessor(X_train)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae, rmse = _metrics_on_price(y_test, pred)
    return EvalResult(name=name, mae=mae, rmse=rmse), pipe


def main():
    data_path = os.environ.get("TEL_AVIV_DATASET", DEFAULT_DATA_PATH)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # target
    target_col = _detect_target(df)

    # time-aware split (если найдётся дата)
    date_col = _detect_date_col(df)
    train_df, test_df = _train_test_time_split(df, date_col=date_col, test_ratio=0.2)

    # features: всё, кроме target
    feature_cols = [c for c in df.columns if c != target_col]

    # y в log1p
    y_train = np.log1p(train_df[target_col].astype(float).values)
    y_test = np.log1p(test_df[target_col].astype(float).values)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # candidates
    candidates = [
        ("RandomForest", RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=800,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )),
        ("HistGB", HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=42
        )),
    ]

    results: List[EvalResult] = []
    trained: Dict[str, Any] = {}

    for name, model in candidates:
        res, pipe = evaluate_candidate(name, model, X_train, y_train, X_test, y_test)
        results.append(res)
        trained[name] = pipe
        print(f"[{name}] MAE={res.mae:,.0f} | RMSE={res.rmse:,.0f}")

    best = sorted(results, key=lambda r: (r.mae, r.rmse))[0]
    best_pipe = trained[best.name]

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    joblib.dump(best_pipe, OUT_MODEL_PATH)

    metrics = {
        "version": "v4_best",
        "best_model": best.name,
        "test_split": "last_20_percent_time_aware" if date_col else "last_20_percent_index",
        "date_col_used": date_col,
        "target_col": target_col,
        "mae": best.mae,
        "rmse": best.rmse,
        "candidates": [{"name": r.name, "mae": r.mae, "rmse": r.rmse} for r in results],
        "note": "MAE/RMSE computed in original price units using expm1(log1p(target))."
    }

    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(OUT_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print("  model   :", OUT_MODEL_PATH)
    print("  metrics :", OUT_METRICS_PATH)
    print("  feats   :", OUT_FEATURES_PATH)
    print(f"\n🏆 Best: {best.name} | MAE={best.mae:,.0f} | RMSE={best.rmse:,.0f}")


if __name__ == "__main__":
    main()
