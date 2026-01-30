from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String

from .db import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_id = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    p10 = Column(Float, nullable=False)
    p50 = Column(Float, nullable=False)
    p90 = Column(Float, nullable=False)
    features = Column(JSON, nullable=False)
    factors = Column(JSON, nullable=False)
    actual_price = Column(Float, nullable=True)
    abs_error = Column(Float, nullable=True)
    pct_error = Column(Float, nullable=True)
