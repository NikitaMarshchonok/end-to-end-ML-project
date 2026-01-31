import os
import time
from typing import Dict

import requests


FX_BASE_URL = os.environ.get("FX_BASE_URL", "https://api.exchangerate.host/latest")
FX_REFRESH_SECONDS = int(os.environ.get("FX_REFRESH_SECONDS", "21600"))  # 6 hours
ALLOWED_CURRENCIES = {"USD", "EUR", "ILS", "TWD"}

_CACHE: Dict[str, dict] = {}


def _fetch_rates(base: str) -> dict:
    params = {"base": base, "symbols": ",".join(sorted(ALLOWED_CURRENCIES))}
    resp = requests.get(FX_BASE_URL, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    if not data or "rates" not in data:
        raise ValueError("Invalid FX response.")
    return data


def get_rates(base: str) -> dict:
    base = base.upper()
    if base not in ALLOWED_CURRENCIES:
        raise ValueError("Unsupported currency.")

    cached = _CACHE.get(base)
    now = time.time()
    if cached and now - cached["ts"] < FX_REFRESH_SECONDS:
        return cached["data"]

    data = _fetch_rates(base)
    _CACHE[base] = {"ts": now, "data": data}
    return data


def convert(amount: float, from_currency: str, to_currency: str) -> float:
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency == to_currency:
        return float(amount)
    rates = get_rates(from_currency).get("rates", {})
    if to_currency not in rates:
        raise ValueError("Target currency not available.")
    return float(amount) * float(rates[to_currency])
