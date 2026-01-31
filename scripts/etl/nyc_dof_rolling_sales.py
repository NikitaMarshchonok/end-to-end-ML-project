import argparse
import csv
from typing import Dict, Any, Iterable

from .common import to_float, to_int, sqft_to_m2, to_date, write_csv


CONTRACT_FIELDS = [
    "market_id",
    "country",
    "city",
    "currency",
    "price",
    "area_m2",
    "transaction_date",
    "source",
    "gross_area_m2",
    "rooms",
    "floor",
    "floors",
    "construction_year",
    "lat",
    "long",
    "property_type",
    "neighborhood",
    "address",
    "building_class",
]


def transform_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    price = to_float(row.get("SALE PRICE"))
    if price is None or price <= 0:
        return None

    area_m2 = sqft_to_m2(row.get("GROSS SQUARE FEET"))
    if area_m2 is None or area_m2 <= 0:
        return None

    out = {
        "market_id": "us-nyc",
        "country": "United States",
        "city": "New York",
        "currency": "USD",
        "price": price,
        "area_m2": area_m2,
        "transaction_date": to_date(row.get("SALE DATE")),
        "source": "nyc_dof_rolling_sales",
        "gross_area_m2": sqft_to_m2(row.get("GROSS SQUARE FEET")),
        "rooms": None,
        "floor": None,
        "floors": None,
        "construction_year": to_int(row.get("YEAR BUILT")),
        "lat": None,
        "long": None,
        "property_type": row.get("BUILDING CLASS CATEGORY"),
        "neighborhood": row.get("NEIGHBORHOOD"),
        "address": row.get("ADDRESS"),
        "building_class": row.get("BUILDING CLASS CATEGORY"),
    }
    return out


def transform_csv(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows: Iterable[Dict[str, Any]] = (transform_row(r) for r in reader)
        filtered = (r for r in rows if r is not None)
        write_csv(output_path, filtered, CONTRACT_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV from NYC DOF Rolling Sales")
    parser.add_argument("--output", required=True, help="Output standardized CSV")
    args = parser.parse_args()
    transform_csv(args.input, args.output)


if __name__ == "__main__":
    main()
