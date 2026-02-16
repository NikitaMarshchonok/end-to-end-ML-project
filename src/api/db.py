import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker


DEFAULT_DB_URL = "sqlite:///./data/predictions.db"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_predictions_table()


def _migrate_sqlite_predictions_table() -> None:
    # Lightweight migration for local SQLite development.
    if not DATABASE_URL.startswith("sqlite"):
        return

    with engine.begin() as conn:
        insp = inspect(conn)
        if "predictions" not in insp.get_table_names():
            return

        existing_cols = {col["name"] for col in insp.get_columns("predictions")}
        required_cols = {
            "market_id": "TEXT DEFAULT 'il-tlv' NOT NULL",
            "area_unit": "TEXT DEFAULT 'm2' NOT NULL",
            "actual_currency": "TEXT",
            "actual_price": "FLOAT",
            "abs_error": "FLOAT",
            "pct_error": "FLOAT",
        }

        for col_name, col_type in required_cols.items():
            if col_name not in existing_cols:
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"))


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
