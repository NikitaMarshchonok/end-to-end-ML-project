import argparse
import os
import sys
from typing import Dict, Any, Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etl.common import to_float, to_int, sqft_to_m2, to_date, write_csv, read_rows, get_value  # noqa: E402


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
    price = to_float(get_value(row, "SALE PRICE"))
    if price is None or price <= 0:
        return None

    area_m2 = sqft_to_m2(get_value(row, "GROSS SQUARE FEET"))
    if area_m2 is None or area_m2 <= 0:
        return None

    out = {
        "market_id": "us-nyc",
        "country": "United States",
        "city": "New York",
        "currency": "USD",
        "price": price,
        "area_m2": area_m2,
        "transaction_date": to_date(get_value(row, "SALE DATE")),
        "source": "nyc_dof_rolling_sales",
        "gross_area_m2": sqft_to_m2(get_value(row, "GROSS SQUARE FEET")),
        "rooms": None,
        "floor": None,
        "floors": None,
        "construction_year": to_int(get_value(row, "YEAR BUILT")),
        "lat": None,
        "long": None,
        "property_type": get_value(row, "BUILDING CLASS CATEGORY"),
        "neighborhood": get_value(row, "NEIGHBORHOOD"),
        "address": get_value(row, "ADDRESS"),
        "building_class": get_value(row, "BUILDING CLASS CATEGORY"),
    }
    return out


def transform_csv(input_path: str, output_path: str) -> None:
    rows = read_rows(input_path)
    parsed: Iterable[Dict[str, Any]] = (transform_row(r) for r in rows)
    filtered = (r for r in parsed if r is not None)
    write_csv(output_path, filtered, CONTRACT_FIELDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV from NYC DOF Rolling Sales")
    parser.add_argument("--output", required=True, help="Output standardized CSV")
    args = parser.parse_args()
    transform_csv(args.input, args.output)


if __name__ == "__main__":
    main()
