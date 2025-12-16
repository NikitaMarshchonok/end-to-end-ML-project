from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_model_artifacts_exist():
    models = ROOT / "models"

    assert (models / "tel_aviv_real_estate_model_v3_2_clean_cli.pkl").exists()
    assert (models / "tel_aviv_metrics_v3_2_clean_cli.json").exists()
    assert (models / "tel_aviv_feature_cols_v3_2_clean_cli.json").exists()

def test_perm_importance_plot_exists():
    pics = ROOT / "pics"
    # если файл у тебя называется чуть иначе — поправь имя тут
    assert (pics / "tel_aviv_v3_2_perm_importance.png").exists() or (pics / "tel_aviv_v3_2_perm_importance.png").exists()
