import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .db import get_session, init_db
from .models import Prediction
from .fx import ALLOWED_CURRENCIES, convert
from ..ml.predictor import get_baseline_stats, get_comparables, list_markets, list_models, predict

AREA_FIELDS = ["netArea", "grossArea", "roof", "yard"]
AREA_UNIT_M2 = "m2"
AREA_UNIT_SQFT = "sqft"
SQFT_TO_M2 = 0.092903


def normalize_features(features: dict[str, Any], area_unit: str) -> dict[str, Any]:
    if area_unit not in (AREA_UNIT_M2, AREA_UNIT_SQFT):
        raise ValueError("area_unit must be 'm2' or 'sqft'.")

    if area_unit == AREA_UNIT_M2:
        return features

    normalized = dict(features)
    for key in AREA_FIELDS:
        if key in normalized and normalized[key] not in [None, ""]:
            try:
                normalized[key] = float(normalized[key]) * SQFT_TO_M2
            except Exception:
                pass
    return normalized


class PredictionRequest(BaseModel):
    market_id: str = Field(..., description="Market identifier")
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")
    area_unit: str = Field("m2", description="Area unit: m2 or sqft")
    display_currency: str | None = Field(None, description="Display currency")


class ExplainRequest(BaseModel):
    market_id: str = Field(..., description="Market identifier")
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")
    area_unit: str = Field("m2", description="Area unit: m2 or sqft")
    display_currency: str | None = Field(None, description="Display currency")


class ComparablesRequest(BaseModel):
    market_id: str = Field(..., description="Market identifier")
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")
    top_k: int = Field(5, ge=1, le=10, description="Number of comparable items")
    area_unit: str = Field("m2", description="Area unit: m2 or sqft")
    display_currency: str | None = Field(None, description="Display currency")


class FeedbackRequest(BaseModel):
    prediction_id: int = Field(..., description="Prediction record id")
    actual_price: float = Field(..., gt=0, description="Observed real price")
    actual_currency: str | None = Field(None, description="Currency of actual_price")


app = FastAPI(title="Real Estate ML API", version="1.0.0")

allow_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
origins = [o.strip() for o in allow_origins.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_READY = False


@app.on_event("startup")
def on_startup():
    global DB_READY
    try:
        init_db()
        DB_READY = True
    except Exception:
        DB_READY = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models(market_id: str | None = None):
    return list_models(market_id)


@app.get("/markets")
def get_markets():
    return list_markets()


@app.post("/predict")
def predict_price(payload: PredictionRequest, db: Session = Depends(get_session)):
    try:
        norm_features = normalize_features(payload.features, payload.area_unit)
        result = predict(payload.model_id, norm_features, payload.market_id)
        display_currency = payload.display_currency.upper() if payload.display_currency else None
        if display_currency and display_currency not in ALLOWED_CURRENCIES:
            raise ValueError("display_currency is not supported.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e

    prediction_id = None
    if DB_READY:
        try:
            record = Prediction(
                market_id=payload.market_id,
                model_id=payload.model_id,
                model_version=result.model_version,
                currency=result.currency,
                price=result.price,
                p10=result.p10,
                p50=result.p50,
                p90=result.p90,
                area_unit=payload.area_unit,
                features=norm_features,
                factors=result.factors,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            prediction_id = record.id
        except SQLAlchemyError:
            db.rollback()

    return {
        "price": result.price,
        "p10": result.p10,
        "p50": result.p50,
        "p90": result.p90,
        "model_version": result.model_version,
        "currency": result.currency,
        "factors": result.factors,
        "prediction_id": prediction_id,
        "display_currency": display_currency,
        "display_price": convert(result.price, result.currency, display_currency)
        if display_currency and display_currency != result.currency
        else None,
        "display_p10": convert(result.p10, result.currency, display_currency)
        if display_currency and display_currency != result.currency
        else None,
        "display_p50": convert(result.p50, result.currency, display_currency)
        if display_currency and display_currency != result.currency
        else None,
        "display_p90": convert(result.p90, result.currency, display_currency)
        if display_currency and display_currency != result.currency
        else None,
    }


@app.post("/explain")
def explain(payload: ExplainRequest):
    try:
        norm_features = normalize_features(payload.features, payload.area_unit)
        result = predict(payload.model_id, norm_features, payload.market_id)
        display_currency = payload.display_currency.upper() if payload.display_currency else None
        if display_currency and display_currency not in ALLOWED_CURRENCIES:
            raise ValueError("display_currency is not supported.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain error: {e}") from e

    display_ranges = None
    if display_currency and display_currency != result.currency:
        display_ranges = [
            {
                "label": label,
                "low": convert(low, result.currency, display_currency),
                "high": convert(high, result.currency, display_currency),
            }
            for (label, low, high) in result.ranges
        ]

    return {
        "model_version": result.model_version,
        "currency": result.currency,
        "factors": result.factors,
        "ranges": [
            {"label": label, "low": low, "high": high}
            for (label, low, high) in result.ranges
        ],
        "display_currency": display_currency,
        "display_ranges": display_ranges,
    }


@app.post("/comparables")
def comparables(payload: ComparablesRequest):
    try:
        display_currency = payload.display_currency.upper() if payload.display_currency else None
        if display_currency and display_currency not in ALLOWED_CURRENCIES:
            raise ValueError("display_currency is not supported.")

        data = get_comparables(
            payload.model_id,
            normalize_features(payload.features, payload.area_unit),
            top_k=payload.top_k,
            market_id=payload.market_id,
        )
        if display_currency and display_currency != data.get("currency"):
            data["items"] = [
                {**item, "price": convert(item["price"], data["currency"], display_currency)}
                for item in data.get("items", [])
            ]
            data["currency"] = display_currency
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparables error: {e}") from e


@app.post("/feedback")
def feedback(payload: FeedbackRequest, db: Session = Depends(get_session)):
    if not DB_READY:
        raise HTTPException(status_code=400, detail="Database is not configured.")

    record = db.query(Prediction).filter(Prediction.id == payload.prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    try:
        actual = float(payload.actual_price)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Actual price must be numeric.") from e

    actual_currency = payload.actual_currency.upper() if payload.actual_currency else record.currency
    if actual_currency not in ALLOWED_CURRENCIES:
        raise HTTPException(status_code=400, detail="actual_currency is not supported.")

    actual_converted = (
        convert(actual, actual_currency, record.currency)
        if actual_currency != record.currency
        else actual
    )

    abs_error = abs(record.price - actual_converted)
    pct_error = abs_error / actual_converted if actual_converted > 0 else None

    record.actual_price = actual_converted
    record.abs_error = abs_error
    record.pct_error = pct_error
    record.actual_currency = actual_currency

    try:
        db.add(record)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e

    return {
        "prediction_id": record.id,
        "actual_price": actual_converted,
        "abs_error": abs_error,
        "pct_error": pct_error,
    }


@app.get("/metrics")
def metrics(model_id: str | None = None, market_id: str | None = None, db: Session = Depends(get_session)):
    if not DB_READY:
        return {"count": 0, "mae": None, "mape": None, "rmse": None}

    query = db.query(Prediction).filter(Prediction.actual_price.isnot(None))
    if model_id:
        query = query.filter(Prediction.model_id == model_id)
    if market_id:
        query = query.filter(Prediction.market_id == market_id)

    rows = query.all()
    if not rows:
        return {"count": 0, "mae": None, "mape": None, "rmse": None}

    abs_errors = [r.abs_error for r in rows if r.abs_error is not None]
    pct_errors = [r.pct_error for r in rows if r.pct_error is not None]
    sq_errors = [r.abs_error ** 2 for r in rows if r.abs_error is not None]

    mae = float(sum(abs_errors) / len(abs_errors)) if abs_errors else None
    mape = float(sum(pct_errors) / len(pct_errors)) if pct_errors else None
    rmse = float((sum(sq_errors) / len(sq_errors)) ** 0.5) if sq_errors else None

    return {
        "count": len(abs_errors),
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
    }


@app.get("/metrics/timeseries")
def metrics_timeseries(
    model_id: str | None = None,
    market_id: str | None = None,
    bucket: str = "day",
    db: Session = Depends(get_session),
):
    if not DB_READY:
        return []

    if bucket not in ("day", "week"):
        raise HTTPException(status_code=400, detail="bucket must be 'day' or 'week'.")

    query = db.query(Prediction).filter(Prediction.actual_price.isnot(None))
    if model_id:
        query = query.filter(Prediction.model_id == model_id)
    if market_id:
        query = query.filter(Prediction.market_id == market_id)

    rows = query.all()
    if not rows:
        return []

    buckets: dict[str, dict] = {}
    for row in rows:
        key = row.created_at.date().isoformat()
        if bucket == "week":
            iso_year, iso_week, _ = row.created_at.isocalendar()
            key = f"{iso_year}-W{iso_week:02d}"

        stats = buckets.setdefault(
            key,
            {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0, "pct_sum": 0.0, "pct_count": 0},
        )
        if row.abs_error is None:
            continue
        stats["count"] += 1
        stats["abs_sum"] += float(row.abs_error)
        stats["sq_sum"] += float(row.abs_error) ** 2
        if row.pct_error is not None:
            stats["pct_sum"] += float(row.pct_error)
            stats["pct_count"] += 1

    out = []
    for key in sorted(buckets.keys()):
        stats = buckets[key]
        if stats["count"] == 0:
            continue
        mae = stats["abs_sum"] / stats["count"]
        rmse = (stats["sq_sum"] / stats["count"]) ** 0.5
        mape = stats["pct_sum"] / stats["pct_count"] if stats["pct_count"] else None
        out.append(
            {
                "bucket": key,
                "count": stats["count"],
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
            }
        )

    return out


@app.get("/predictions")
def list_predictions(limit: int = 10, db: Session = Depends(get_session)):
    if not DB_READY:
        return []

    rows = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return [
        {
            "id": row.id,
            "created_at": row.created_at.isoformat(),
            "market_id": row.market_id,
            "model_id": row.model_id,
            "model_version": row.model_version,
            "currency": row.currency,
            "price": row.price,
            "p10": row.p10,
            "p50": row.p50,
            "p90": row.p90,
            "area_unit": row.area_unit,
            "features": row.features,
            "factors": row.factors,
        }
        for row in rows
    ]


@app.delete("/predictions")
def clear_predictions(db: Session = Depends(get_session)):
    if not DB_READY:
        raise HTTPException(status_code=400, detail="Database is not configured.")
    try:
        db.query(Prediction).delete()
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}") from e

    return {"status": "ok"}


@app.get("/monitoring")
def monitoring(model_id: str, market_id: str | None = None, limit: int = 500, db: Session = Depends(get_session)):
    if not DB_READY:
        return {
            "model_id": model_id,
            "total_predictions": 0,
            "sample_size": 0,
            "drift": [],
            "note": "Database is not configured.",
        }

    query = db.query(Prediction).filter(Prediction.model_id == model_id)
    if market_id:
        query = query.filter(Prediction.market_id == market_id)

    rows = (
        query.order_by(Prediction.created_at.desc())
        .limit(min(limit, 2000))
        .all()
    )
    if not rows:
        return {
            "model_id": model_id,
            "total_predictions": 0,
            "sample_size": 0,
            "drift": [],
            "note": "No prediction history yet.",
        }

    baseline = get_baseline_stats(model_id, market_id=market_id) or {}
    if not baseline:
        return {
            "model_id": model_id,
            "total_predictions": len(rows),
            "sample_size": len(rows),
            "drift": [],
            "note": "Baseline data not available.",
        }

    # Build recent feature table
    features_list = [row.features for row in rows if isinstance(row.features, dict)]
    drift = []
    for feature, stats in baseline.items():
        vals = []
        for item in features_list:
            val = item.get(feature)
            try:
                if val not in [None, ""]:
                    vals.append(float(val))
            except Exception:
                continue

        if not vals:
            continue

        recent_mean = float(sum(vals) / len(vals))
        std = float(stats.get("std") or 1.0)
        baseline_mean = float(stats.get("mean") or 0.0)
        drift_score = abs(recent_mean - baseline_mean) / std if std > 0 else 0.0

        drift.append(
            {
                "feature": feature,
                "baseline_mean": baseline_mean,
                "recent_mean": recent_mean,
                "baseline_std": std,
                "drift_score": drift_score,
                "sample_size": len(vals),
            }
        )

    drift.sort(key=lambda x: x.get("drift_score", 0), reverse=True)

    return {
        "model_id": model_id,
        "total_predictions": len(rows),
        "sample_size": len(rows),
        "drift": drift,
    }
