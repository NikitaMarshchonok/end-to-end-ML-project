import csv
from datetime import datetime
from typing import Any, Dict, Iterable, List


SQFT_TO_M2 = 0.092903


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    f = to_float(value)
    if f is None:
        return None
    return int(f)


def to_date(value: Any, fmt: str | None = None) -> str | None:
    if value in (None, ""):
        return None
    try:
        if fmt:
            return datetime.strptime(str(value), fmt).date().isoformat()
        return datetime.fromisoformat(str(value)).date().isoformat()
    except Exception:
        return None


def sqft_to_m2(value: Any) -> float | None:
    f = to_float(value)
    if f is None:
        return None
    return f * SQFT_TO_M2


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
