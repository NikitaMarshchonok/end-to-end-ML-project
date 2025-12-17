from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def pick_metric(data: dict, candidates: list[str]):
    # 1) exact match
    for k in candidates:
        if k in data:
            return data[k]

    # 2) case-insensitive exact
    lower_map = {k.lower(): k for k in data.keys()}
    for k in candidates:
        kk = k.lower()
        if kk in lower_map:
            return data[lower_map[kk]]

    # 3) substring match (e.g. "mae_nis", "RMSE_NIS")
    for key in data.keys():
        key_l = key.lower()
        for cand in candidates:
            if cand.lower() in key_l:
                return data[key]

    raise AssertionError(
        f"Can't find metric keys {candidates}. Available keys: {sorted(data.keys())}"
    )

def test_metrics_json_has_expected_keys():
    p = ROOT / "models" / "tel_aviv_metrics_v3_2_clean_cli.json"
    data = json.loads(p.read_text())

    mae = pick_metric(data, ["mae", "mae_nis", "mae (nis)"])
    rmse = pick_metric(data, ["rmse", "rmse_nis", "rmse (nis)"])
    r2 = pick_metric(data, ["r2", "r_2", "r2_score"])

    assert isinstance(mae, (int, float))
    assert isinstance(rmse, (int, float))
    assert isinstance(r2, (int, float))
