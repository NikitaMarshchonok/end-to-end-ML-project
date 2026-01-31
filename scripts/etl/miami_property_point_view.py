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
    price = to_float(row.get("SALE_AMT") or row.get("SALE_AMOUNT") or row.get("SALEPRICE"))
    if price is None or price <= 0:
        return None

    area_m2 = sqft_to_m2(row.get("LIVING_AREA") or row.get("LIVING_AREA_SQFT") or row.get("SQFT"))
    if area_m2 is None or area_m2 <= 0:
        return None

    out = {
        "market_id": "us-mia",
        "country": "United States",
        "city": "Miami",
        "currency": "USD",
        "price": price,
        "area_m2": area_m2,
        "transaction_date": to_date(row.get("SALE_DATE") or row.get("DATE_OF_SALE")),
        "source": "miami_dade_property_point_view",
        "gross_area_m2": area_m2,
        "rooms": to_float(row.get("BEDROOMS")),
        "floor": None,
        "floors": None,
        "construction_year": to_int(row.get("YEAR_BUILT")),
        "lat": to_float(row.get("LAT") or row.get("LATITUDE")),
        "long": to_float(row.get("LON") or row.get("LONG") or row.get("LONGITUDE")),
        "property_type": row.get("PROPERTY_TYPE"),
        "neighborhood": row.get("NEIGHBORHOOD"),
        "address": row.get("SITE_ADDRESS") or row.get("ADDRESS"),
        "building_class": row.get("CLASS"),
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
    parser.add_argument("--input", required=True, help="Input CSV from Miami-Dade Property Appraiser")
    parser.add_argument("--output", required=True, help="Output standardized CSV")
    args = parser.parse_args()
    transform_csv(args.input, args.output)


if __name__ == "__main__":
    main()
