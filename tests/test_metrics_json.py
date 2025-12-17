import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_metrics_json_has_expected_keys():
    p = ROOT / "models" / "tel_aviv_metrics_v3_2_clean_cli.json"
    data = json.loads(p.read_text())

    # ожидаемые ключи (если у тебя другие — скажи, подстрою)
    assert "mae" in data
    assert "rmse" in data
    assert "r2" in data

    assert data["mae"] > 0
