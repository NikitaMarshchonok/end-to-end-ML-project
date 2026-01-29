import os
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .db import get_session, init_db
from .models import Prediction
from ..ml.predictor import get_baseline_stats, get_comparables, list_models, predict


class PredictionRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")


class ExplainRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")


class ComparablesRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    features: dict[str, Any] = Field(..., description="Feature values")
    top_k: int = Field(5, ge=1, le=10, description="Number of comparable items")


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
def get_models():
    return list_models()


@app.post("/predict")
def predict_price(payload: PredictionRequest, db: Session = Depends(get_session)):
    try:
        result = predict(payload.model_id, payload.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e

    if DB_READY:
        try:
            record = Prediction(
                model_id=payload.model_id,
                model_version=result.model_version,
                currency=result.currency,
                price=result.price,
                p10=result.p10,
                p50=result.p50,
                p90=result.p90,
                features=payload.features,
                factors=result.factors,
            )
            db.add(record)
            db.commit()
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
    }


@app.post("/explain")
def explain(payload: ExplainRequest):
    try:
        result = predict(payload.model_id, payload.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain error: {e}") from e

    return {
        "model_version": result.model_version,
        "currency": result.currency,
        "factors": result.factors,
        "ranges": [
            {"label": label, "low": low, "high": high}
            for (label, low, high) in result.ranges
        ],
    }


@app.post("/comparables")
def comparables(payload: ComparablesRequest):
    try:
        return get_comparables(payload.model_id, payload.features, top_k=payload.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparables error: {e}") from e


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
            "model_id": row.model_id,
            "model_version": row.model_version,
            "currency": row.currency,
            "price": row.price,
            "p10": row.p10,
            "p50": row.p50,
            "p90": row.p90,
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
def monitoring(model_id: str, limit: int = 500, db: Session = Depends(get_session)):
    if not DB_READY:
        return {
            "model_id": model_id,
            "total_predictions": 0,
            "sample_size": 0,
            "drift": [],
            "note": "Database is not configured.",
        }

    rows = (
        db.query(Prediction)
        .filter(Prediction.model_id == model_id)
        .order_by(Prediction.created_at.desc())
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

    baseline = get_baseline_stats(model_id) or {}
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
