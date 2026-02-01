import csv
import os
from typing import Iterable, Dict, Any, List

import pandas as pd
from datetime import datetime
from typing import Any, Dict, Iterable, List


SQFT_TO_M2 = 0.092903


def to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        cleaned = str(value).strip().replace(",", "")
        cleaned = cleaned.replace("$", "")
        return float(cleaned)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    f = to_float(value)
    if f is None:
        return None
    try:
        if f != f:
            return None
        return int(f)
    except Exception:
        return None


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(input_path: str) -> Iterable[Dict[str, Any]]:
    ext = os.path.splitext(input_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # Rolling sales files often have header rows not on the first line.
        preview = pd.read_excel(input_path, header=None)
        header_row = None
        for idx, row in preview.iterrows():
            values = [str(v).strip().lower() for v in row.values if v is not None]
            if any(v == "sale price" for v in values):
                header_row = idx
                break
        if header_row is not None:
            df = pd.read_excel(input_path, header=header_row)
        else:
            df = pd.read_excel(input_path)
        return df.to_dict(orient="records")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_value(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
        # try case-insensitive / trimmed match
        for k in row.keys():
            if k.strip().lower() == key.strip().lower():
                return row.get(k)
    return None
