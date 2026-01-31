# Real Estate Data Contract (v1)

This contract defines the **standardized schema** used by all markets.
Raw datasets must be mapped into this schema before training or inference.

## Required fields
- `market_id` (string) — e.g. `il-tlv`, `us-nyc`, `us-mia`
- `country` (string) — ISO short name (e.g. `Israel`, `United States`)
- `city` (string)
- `currency` (string) — ISO 4217 currency code (e.g. `ILS`, `USD`)
- `price` (number) — sale price in `currency`
- `area_m2` (number) — net or living area in **square meters**
- `transaction_date` (date) — `YYYY-MM-DD`
- `source` (string) — dataset/source identifier

## Optional fields
- `gross_area_m2` (number)
- `rooms` (number)
- `floor` (number)
- `floors` (number)
- `construction_year` (number)
- `lat` (number)
- `long` (number)
- `property_type` (string) — mapped to canonical values if possible
- `neighborhood` (string)
- `address` (string)
- `building_class` (string)

## Notes
- If raw area is in sqft, convert using `1 sqft = 0.092903 m²`.
- If currency differs by market, keep `price` in local currency and set `currency`.
- For ML training, you may derive features from `transaction_date` and `construction_year`.
