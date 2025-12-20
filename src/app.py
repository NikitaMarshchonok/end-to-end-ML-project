import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


# =========================================================
# Build tag (чтобы ты видел, что запустился ИМЕННО этот файл)
# =========================================================
APP_BUILD = "UI_v3_fixed_css_injected__2025-12-20"


# =========================================================
# Helpers
# =========================================================
def first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def safe_load_json(path: str) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =========================================================
# Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Taiwan model (учтём регистр на Linux)
MODEL_TAIWAN_PATH = first_existing(
    os.path.join(MODELS_DIR, "real_estate_model.pkl"),
    os.path.join(MODELS_DIR, "Real_estate_model.pkl"),
)

# Tel Aviv v1
MODEL_TEL_AVIV_V1_PATH = os.path.join(MODELS_DIR, "tel_aviv_real_estate_model.pkl")

# Tel Aviv v2
MODEL_TEL_AVIV_V2_PATH = os.path.join(MODELS_DIR, "tel_aviv_real_estate_model_v2.pkl")
TEL_AVIV_V2_FEATS_PATH = os.path.join(MODELS_DIR, "tel_aviv_feature_cols_v2.json")

# Tel Aviv v3.2 clean (CLI artifacts)
MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_real_estate_model_v3_2_clean_cli.pkl"
)
TEL_AVIV_V3_2_FEATS_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_feature_cols_v3_2_clean_cli.json"
)
TEL_AVIV_V3_2_METRICS_PATH = os.path.join(
    MODELS_DIR, "tel_aviv_metrics_v3_2_clean_cli.json"
)


# =========================================================
# Load models (не падаем, если чего-то нет — просто скрываем опции)
# =========================================================
model_taiwan = None
if MODEL_TAIWAN_PATH:
    try:
        model_taiwan = joblib.load(MODEL_TAIWAN_PATH)
    except Exception:
        model_taiwan = None

model_tel_aviv_v1 = None
if os.path.exists(MODEL_TEL_AVIV_V1_PATH):
    try:
        model_tel_aviv_v1 = joblib.load(MODEL_TEL_AVIV_V1_PATH)
    except Exception:
        model_tel_aviv_v1 = None

model_tel_aviv_v2 = None
tel_aviv_v2_feature_cols = None
if os.path.exists(MODEL_TEL_AVIV_V2_PATH):
    try:
        model_tel_aviv_v2 = joblib.load(MODEL_TEL_AVIV_V2_PATH)
    except Exception:
        model_tel_aviv_v2 = None
    tel_aviv_v2_feature_cols = safe_load_json(TEL_AVIV_V2_FEATS_PATH)

model_tel_aviv_v3_2 = None
tel_aviv_v3_2_feature_cols = None
tel_aviv_v3_2_metrics = None
if os.path.exists(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH):
    try:
        model_tel_aviv_v3_2 = joblib.load(MODEL_TEL_AVIV_V3_2_CLEAN_CLI_PATH)
    except Exception:
        model_tel_aviv_v3_2 = None
    tel_aviv_v3_2_feature_cols = safe_load_json(TEL_AVIV_V3_2_FEATS_PATH)
    tel_aviv_v3_2_metrics = safe_load_json(TEL_AVIV_V3_2_METRICS_PATH)


# =========================================================
# Metrics / Model cards (портфолио-стиль)
# =========================================================
def pick_metric(metrics: dict | None, key: str, fallback):
    if isinstance(metrics, dict) and key in metrics and metrics[key] is not None:
        return metrics[key]
    return fallback


MODEL_CARDS = {
    "tel_aviv_v3_2_clean": {
        "title": "Tel Aviv model v3.2_clean (BEST — time split + cleaning)",
        "mae": pick_metric(tel_aviv_v3_2_metrics, "mae", 928_568),
        "rmse": pick_metric(tel_aviv_v3_2_metrics, "rmse", 1_503_636),
        "r2": pick_metric(tel_aviv_v3_2_metrics, "r2", 0.575),
        "note": "RandomForest tuned + missing indicators. Time-aware split after cleaning.",
        "pill": "BEST",
    },
    "tel_aviv_v2": {
        "title": "Tel Aviv model v2 (RandomForest, log(price))",
        "mae": 669_657,
        "rmse": 1_418_990,
        "r2": 0.610,
        "note": "Engineered features + log(price).",
        "pill": "IMPROVED",
    },
    "tel_aviv_v1": {
        "title": "Tel Aviv model v1 (baseline)",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Baseline with 4 raw inputs.",
        "pill": "BASELINE",
    },
    "taiwan": {
        "title": "Taiwan tutorial model",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Classic tutorial dataset for end-to-end demo.",
        "pill": "TUTORIAL",
    },
}


# =========================================================
# Fallback columns
# =========================================================
TEL_AVIV_V2_FALLBACK_COLS = [
    "netArea",
    "grossArea",
    "rooms",
    "floor",
    "floors",
    "apartmentsInBuilding",
    "parking",
    "storage",
    "roof",
    "yard",
    "constructionYear",
    "tx_year",
    "tx_month",
    "tx_quarter",
    "building_age_at_tx",
    "floor_ratio",
]


# =========================================================
# Feature builder (v2/v3.2 share same schema)
# =========================================================
def build_tel_aviv_features(
    netarea, rooms, floor, year,
    gross_area=None, floors=None,
    apartments_in_building=None,
    parking=None, storage=None, roof=None, yard=None,
    feature_cols_override=None
):
    netarea = float(netarea)
    rooms = float(rooms)
    floor = float(floor)
    year = float(year)

    gross_area = float(gross_area) if gross_area not in [None, ""] else np.nan
    floors = float(floors) if floors not in [None, ""] else np.nan
    apartments_in_building = float(apartments_in_building) if apartments_in_building not in [None, ""] else np.nan
    parking = float(parking) if parking not in [None, ""] else np.nan
    storage = float(storage) if storage not in [None, ""] else np.nan
    roof = float(roof) if roof not in [None, ""] else np.nan
    yard = float(yard) if yard not in [None, ""] else np.nan

    if np.isnan(gross_area):
        gross_area = netarea

    now = datetime.now()
    tx_year = now.year
    tx_month = now.month
    tx_quarter = (tx_month - 1) // 3 + 1

    building_age_at_tx = np.nan
    if year and year > 0:
        building_age_at_tx = max(0, tx_year - int(year))

    floor_ratio = np.nan
    if not np.isnan(floors) and floors > 0:
        floor_ratio = floor / floors

    row = {
        "netArea": netarea,
        "grossArea": gross_area,
        "rooms": rooms,
        "floor": floor,
        "floors": floors,
        "apartmentsInBuilding": apartments_in_building,
        "parking": parking,
        "storage": storage,
        "roof": roof,
        "yard": yard,
        "constructionYear": year,
        "tx_year": tx_year,
        "tx_month": tx_month,
        "tx_quarter": tx_quarter,
        "building_age_at_tx": building_age_at_tx,
        "floor_ratio": floor_ratio,
    }

    feature_cols = (
        feature_cols_override
        or tel_aviv_v3_2_feature_cols
        or tel_aviv_v2_feature_cols
        or TEL_AVIV_V2_FALLBACK_COLS
    )

    X = pd.DataFrame([row])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_cols]


def validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=None):
    try:
        netarea_f = float(netarea)
        rooms_f = float(rooms)
        floor_f = float(floor)
        year_f = float(year)
    except Exception:
        return "Please enter valid numeric values for required Tel Aviv fields."

    if netarea_f <= 0 or netarea_f > 1000:
        return "Net area must be > 0 and look realistic (e.g., 20–300 m²)."
    if rooms_f <= 0 or rooms_f > 12:
        return "Rooms must be in a realistic range (e.g., 1–12)."
    if floor_f < -2 or floor_f > 200:
        return "Floor must be in a realistic range (e.g., -2 to 200)."

    current_year = datetime.now().year
    if year_f < 1900 or year_f > current_year + 1:
        return f"Construction year must be between 1900 and {current_year + 1}."

    if floors_total not in [None, ""]:
        try:
            floors_total_f = float(floors_total)
            if floors_total_f <= 0 or floors_total_f > 300:
                return "Total floors must be in a realistic range (1–300)."
            if floor_f > floors_total_f:
                return "Floor cannot be greater than total floors in building."
        except Exception:
            return "Total floors must be numeric."
    return None


# =========================================================
# Dash app + CSS (ВАЖНО: подключаем через app.index_string)
# =========================================================
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Real Estate Price Prediction"

CSS = """
:root{
  --bg:#f6f8fc;
  --card:#ffffff;
  --text:#0b1220;
  --muted:#5b6b85;
  --border:#e3eaf6;
  --shadow: 0 12px 34px rgba(15,23,42,.08);
  --shadow2: 0 6px 18px rgba(15,23,42,.06);
  --primary:#2563eb;
  --primary2:#1d4ed8;
  --danger:#b00020;
  --pill:#eef2ff;
  --ok:#16a34a;
  --warn:#f59e0b;
}
*{box-sizing:border-box}
body{
  margin:0;
  background: radial-gradient(1200px 600px at 20% -10%, #eaf0ff 0%, rgba(234,240,255,0) 60%),
              radial-gradient(1000px 500px at 90% 10%, #f0f7ff 0%, rgba(240,247,255,0) 55%),
              var(--bg);
  color:var(--text);
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}
.container{max-width:1020px;margin:34px auto;padding:0 16px}
.header{text-align:center;margin-bottom:18px}
.h1{font-size:42px;letter-spacing:-0.02em;line-height:1.1;margin:0}
.sub{margin-top:10px;color:var(--muted);font-size:14px}
.badges{margin-top:12px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap}
.pill{
  display:inline-flex;align-items:center;gap:8px;
  font-size:12px;font-weight:800;
  padding:6px 10px;border-radius:999px;
  background:var(--pill);border:1px solid var(--border);color:#22314f;
}
.dot{width:8px;height:8px;border-radius:999px;background:var(--ok)}
.dotWarn{background:var(--warn)}
.card{
  background:rgba(255,255,255,.92);
  border:1px solid var(--border);
  border-radius:18px;
  box-shadow:var(--shadow);
  padding:18px;
  margin-bottom:14px;
  backdrop-filter: blur(6px);
}
.cardTitle{font-weight:900;margin:0 0 12px 0;font-size:14px;color:#1b2b4a;text-transform:uppercase;letter-spacing:.08em}
.hr{height:1px;background:var(--border);margin:12px 0}
.label{font-size:13px;font-weight:800;margin-bottom:6px;display:block;color:#22314f}
.muted{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media (max-width:900px){.grid4{grid-template-columns:1fr 1fr}}
@media (max-width:640px){.grid2{grid-template-columns:1fr}.grid4{grid-template-columns:1fr}}
.input, input{
  width:100%;
  padding:11px 12px;
  border-radius:14px;
  border:1px solid #d3deef;
  outline:none;
  background:#fff;
  box-shadow: var(--shadow2);
}
input::placeholder{color:#97a6bf}
input:focus{
  border-color:rgba(37,99,235,.55);
  box-shadow:0 0 0 4px rgba(37,99,235,.14), var(--shadow2);
}
.btnRow{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
.presetBtn{
  border:1px solid var(--border);
  background:#fff;
  padding:10px 12px;
  border-radius:14px;
  cursor:pointer;
  font-weight:900;
  box-shadow: var(--shadow2);
}
.presetBtn:hover{border-color:rgba(37,99,235,.35);background:#f8fbff}
.primaryBtn{
  width:100%;
  border:none;
  background:linear-gradient(180deg,var(--primary),var(--primary2));
  color:#fff;
  padding:14px 14px;
  border-radius:16px;
  font-weight:950;
  cursor:pointer;
  font-size:16px;
  box-shadow: 0 14px 26px rgba(37,99,235,.18);
}
.primaryBtn:hover{filter:brightness(1.03)}
.error{color:var(--danger);font-size:13px;margin-top:10px;font-weight:800}
.modelCard{
  border-left:6px solid var(--primary);
  background:linear-gradient(180deg,#f8faff,#ffffff);
}
.rowBetween{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.bigTitle{font-weight:950;font-size:18px}
.kpiRow{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px}
.kpi{
  flex:1 1 160px;
  border:1px solid var(--border);
  background:#fff;
  border-radius:16px;
  padding:12px;
  box-shadow: var(--shadow2);
}
.kpi .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:900}
.kpi .v{margin-top:6px;font-size:16px;font-weight:950;color:#0b1220}
.result{
  text-align:center;
  font-size:18px;
  font-weight:950;
  padding:18px;
  border-radius:18px;
  border:1px solid var(--border);
  background:#fff;
  box-shadow:var(--shadow);
}
.smallList{margin:0;padding-left:18px}
.radioWrap label{cursor:pointer}
.radioWrap input{margin-right:8px; transform: translateY(1px);}
.footer{
  text-align:center;
  margin:18px 0 10px 0;
  color:var(--muted);
  font-size:12px;
}
"""

# ✅ ВОТ ЭТО И ЕСТЬ ПРАВИЛЬНОЕ ПОДКЛЮЧЕНИЕ CSS В ОДНОМ ФАЙЛЕ
app.index_string = f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
    <style>{CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""


# =========================================================
# Options
# =========================================================
MODEL_OPTIONS = []
if model_tel_aviv_v3_2 is not None:
    MODEL_OPTIONS.append({"label": "Tel Aviv model v3.2_clean (BEST — time split + cleaning)", "value": "tel_aviv_v3_2_clean"})
if model_tel_aviv_v2 is not None:
    MODEL_OPTIONS.append({"label": "Tel Aviv model v2 (improved, log(price))", "value": "tel_aviv_v2"})
if model_tel_aviv_v1 is not None:
    MODEL_OPTIONS.append({"label": "Tel Aviv model v1 (baseline)", "value": "tel_aviv_v1"})
if model_taiwan is not None:
    MODEL_OPTIONS.append({"label": "Tutorial model (Taiwan dataset)", "value": "taiwan"})

if not MODEL_OPTIONS:
    MODEL_OPTIONS = [{"label": "No models found in /models", "value": "missing"}]


PRESETS = {
    "studio": {"netarea": 35, "rooms": 1.0, "floor": 2, "year": 1995},
    "family": {"netarea": 85, "rooms": 3.0, "floor": 5, "year": 2008},
    "penthouse": {"netarea": 150, "rooms": 4.5, "floor": 18, "year": 2020},
}


# =========================================================
# Layout
# =========================================================
app.layout = html.Div(
    className="container",
    children=[
        html.Div(
            className="header",
            children=[
                html.H1("Real Estate Price Prediction", className="h1"),
                html.Div("Choose a model and fill the corresponding fields.", className="sub"),
                html.Div(
                    className="badges",
                    children=[
                        html.Span([html.Span(className="dot"), " UI ready"], className="pill"),
                        html.Span([html.Span(className="dot"), f" Build: {APP_BUILD}"], className="pill"),
                    ],
                ),
            ],
        ),

        html.Div(
            className="card",
            children=[
                html.Div("Select model", className="cardTitle"),
                dcc.RadioItems(
                    id="model-choice",
                    options=MODEL_OPTIONS,
                    value=MODEL_OPTIONS[0]["value"],
                    className="radioWrap",
                    labelStyle={"display": "block", "marginTop": "10px", "fontWeight": "800"},
                    style={"fontSize": "14px"},
                ),
                html.Div("Tip: v2/v3 models use engineered features and log(price).", className="muted", style={"marginTop": "10px"}),
            ],
        ),

        html.Div(id="model-card", className="card modelCard"),

        # Tel Aviv required
        html.Div(
            id="section-telaviv",
            className="card",
            children=[
                html.Div("Tel Aviv — required inputs", className="cardTitle"),

                html.Div(
                    children=[
                        html.Div("Presets", className="label"),
                        html.Div(
                            className="btnRow",
                            children=[
                                html.Button("1-room studio", id="preset-studio", n_clicks=0, className="presetBtn"),
                                html.Button("3-room family", id="preset-family", n_clicks=0, className="presetBtn"),
                                html.Button("Penthouse", id="preset-penthouse", n_clicks=0, className="presetBtn"),
                            ],
                        ),
                        html.Div("Click a preset to auto-fill realistic values.", className="muted"),
                        html.Div(className="hr"),
                    ]
                ),

                html.Div(
                    className="grid4",
                    children=[
                        html.Div([html.Label("Net area (m²) *", className="label"), dcc.Input(id="input-netarea", type="number", placeholder="e.g. 80", className="input")]),
                        html.Div([html.Label("Rooms *", className="label"), dcc.Input(id="input-rooms", type="number", placeholder="e.g. 3", className="input")]),
                        html.Div([html.Label("Floor *", className="label"), dcc.Input(id="input-floor", type="number", placeholder="e.g. 4", className="input")]),
                        html.Div([html.Label("Construction year *", className="label"), dcc.Input(id="input-year", type="number", placeholder="e.g. 2010", className="input")]),
                    ],
                ),
                html.Div("(* required for Tel Aviv models)", className="muted", style={"marginTop": "10px"}),
            ],
        ),

        # Tel Aviv optional
        html.Div(
            id="section-telaviv-opt",
            className="card",
            children=[
                html.Div("Tel Aviv v2 / v3.2_clean — optional fields", className="cardTitle"),
                html.Div("These can improve accuracy. You can leave them empty.", className="muted"),
                html.Div(className="hr"),

                html.Div(
                    className="grid4",
                    children=[
                        html.Div([html.Label("Gross area (m²)", className="label"), dcc.Input(id="input-grossarea", type="number", placeholder="e.g. 95", className="input")]),
                        html.Div([html.Label("Total floors", className="label"), dcc.Input(id="input-floors", type="number", placeholder="e.g. 12", className="input")]),
                        html.Div([html.Label("Apartments in building", className="label"), dcc.Input(id="input-apts", type="number", placeholder="e.g. 40", className="input")]),
                        html.Div([html.Label("Parking spots", className="label"), dcc.Input(id="input-parking", type="number", placeholder="e.g. 1", className="input")]),
                        html.Div([html.Label("Storage", className="label"), dcc.Input(id="input-storage", type="number", placeholder="e.g. 1", className="input")]),
                        html.Div([html.Label("Roof area", className="label"), dcc.Input(id="input-roof", type="number", placeholder="e.g. 20", className="input")]),
                        html.Div([html.Label("Yard area", className="label"), dcc.Input(id="input-yard", type="number", placeholder="e.g. 30", className="input")]),
                    ],
                ),
            ],
        ),

        # Taiwan
        html.Div(
            id="section-taiwan",
            className="card",
            children=[
                html.Div("Taiwan — inputs", className="cardTitle"),
                html.Div("Use these fields only when Taiwan model is selected.", className="muted"),
                html.Div(className="hr"),

                html.Div(
                    className="grid2",
                    children=[
                        html.Div([html.Label("Distance to MRT", className="label"), dcc.Input(id="input-distance", type="number", placeholder="e.g. 400", className="input")]),
                        html.Div([html.Label("Convenience stores", className="label"), dcc.Input(id="input-convenience", type="number", placeholder="e.g. 4", className="input")]),
                        html.Div([html.Label("Latitude", className="label"), dcc.Input(id="input-lat", type="number", placeholder="e.g. 24.98", className="input")]),
                        html.Div([html.Label("Longitude", className="label"), dcc.Input(id="input-long", type="number", placeholder="e.g. 121.54", className="input")]),
                    ],
                ),
            ],
        ),

        # Action
        html.Div(
            className="card",
            children=[
                html.Button("Predict price", id="predict-button", n_clicks=0, className="primaryBtn"),
                html.Div(id="validation-output", className="error"),
            ],
        ),

        # Result
        html.Div(id="prediction-output", className="result", children=""),

        html.Div("Portfolio demo • Dash UI • Multiple model versions • Time-aware evaluation", className="footer"),
    ],
)


# =========================================================
# Visibility
# =========================================================
@app.callback(
    Output("section-telaviv", "style"),
    Output("section-telaviv-opt", "style"),
    Output("section-taiwan", "style"),
    Input("model-choice", "value"),
)
def toggle_sections(model_choice):
    show_telaviv = model_choice in ("tel_aviv_v1", "tel_aviv_v2", "tel_aviv_v3_2_clean")
    show_opt = model_choice in ("tel_aviv_v2", "tel_aviv_v3_2_clean")
    show_taiwan = model_choice == "taiwan"

    def s(show: bool):
        return {"display": "block"} if show else {"display": "none"}

    return s(show_telaviv), s(show_opt), s(show_taiwan)


# =========================================================
# Button label
# =========================================================
@app.callback(
    Output("predict-button", "children"),
    Input("model-choice", "value"),
)
def update_button_text(model_choice):
    if model_choice == "tel_aviv_v3_2_clean":
        return "Predict with Tel Aviv v3.2_clean (BEST)"
    if model_choice == "tel_aviv_v2":
        return "Predict with Tel Aviv v2"
    if model_choice == "tel_aviv_v1":
        return "Predict with Tel Aviv v1"
    if model_choice == "taiwan":
        return "Predict with Taiwan model"
    return "Predict price"


# =========================================================
# Model Card
# =========================================================
@app.callback(
    Output("model-card", "children"),
    Input("model-choice", "value"),
)
def update_model_card(model_choice):
    card = MODEL_CARDS.get(model_choice)
    if not card:
        return [html.Div("Model info not available.", className="muted")]

    def fmt_money(v):
        return f"{int(v):,} ₪" if v is not None else "—"

    def fmt_num(v):
        return f"{float(v):.3f}" if v is not None else "—"

    pill = card.get("pill", "MODEL")

    return [
        html.Div(
            className="rowBetween",
            children=[
                html.Div("Model card", className="cardTitle", style={"marginBottom": "0"}),
                html.Span([html.Span(className="dot"), f" {pill}"], className="pill"),
            ],
        ),
        html.Div(card["title"], className="bigTitle", style={"marginTop": "10px"}),

        html.Div(
            className="kpiRow",
            children=[
                html.Div([html.Div("MAE", className="k"), html.Div(fmt_money(card.get("mae")), className="v")], className="kpi"),
                html.Div([html.Div("RMSE", className="k"), html.Div(fmt_money(card.get("rmse")), className="v")], className="kpi"),
                html.Div([html.Div("R²", className="k"), html.Div(fmt_num(card.get("r2")), className="v")], className="kpi"),
            ],
        ),

        html.Div(card.get("note", ""), className="muted", style={"marginTop": "12px"}),
    ]


# =========================================================
# Presets
# =========================================================
@app.callback(
    Output("input-netarea", "value"),
    Output("input-rooms", "value"),
    Output("input-floor", "value"),
    Output("input-year", "value"),
    Input("preset-studio", "n_clicks"),
    Input("preset-family", "n_clicks"),
    Input("preset-penthouse", "n_clicks"),
    prevent_initial_call=True,
)
def apply_presets(n1, n2, n3):
    ctx = dash.callback_context
    trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    if trig == "preset-studio":
        p = PRESETS["studio"]
    elif trig == "preset-family":
        p = PRESETS["family"]
    else:
        p = PRESETS["penthouse"]
    return p["netarea"], p["rooms"], p["floor"], p["year"]


# =========================================================
# Predict
# =========================================================
@app.callback(
    Output("prediction-output", "children"),
    Output("validation-output", "children"),
    Input("predict-button", "n_clicks"),
    State("model-choice", "value"),

    # Tel Aviv required
    State("input-netarea", "value"),
    State("input-rooms", "value"),
    State("input-floor", "value"),
    State("input-year", "value"),

    # Tel Aviv optional
    State("input-grossarea", "value"),
    State("input-floors", "value"),
    State("input-apts", "value"),
    State("input-parking", "value"),
    State("input-storage", "value"),
    State("input-roof", "value"),
    State("input-yard", "value"),

    # Taiwan
    State("input-distance", "value"),
    State("input-convenience", "value"),
    State("input-lat", "value"),
    State("input-long", "value"),
)
def predict_price(
    n_clicks, model_choice,
    netarea, rooms, floor, year,
    grossarea, floors_total, apts, parking, storage, roof, yard,
    distance, convenience, lat, long_,
):
    if not n_clicks:
        return "", ""

    # Tel Aviv v3.2
    if model_choice == "tel_aviv_v3_2_clean":
        if model_tel_aviv_v3_2 is None:
            return "", "Tel Aviv v3.2_clean model not available."
        if None in (netarea, rooms, floor, year):
            return "", "Please fill required Tel Aviv fields."
        err = validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=floors_total)
        if err:
            return "", err
        try:
            X = build_tel_aviv_features(
                netarea, rooms, floor, year,
                gross_area=grossarea, floors=floors_total,
                apartments_in_building=apts,
                parking=parking, storage=storage, roof=roof, yard=yard,
                feature_cols_override=tel_aviv_v3_2_feature_cols,
            )
            pred_log = float(model_tel_aviv_v3_2.predict(X)[0])
            price = float(np.expm1(pred_log))
            price = max(price, 0.0)
            return f"Tel Aviv v3.2_clean — estimated price: {price:,.0f} ₪", ""
        except Exception as e:
            return "", f"Error during prediction: {e}"

    # Tel Aviv v2
    if model_choice == "tel_aviv_v2":
        if model_tel_aviv_v2 is None:
            return "", "Tel Aviv v2 model not available."
        if None in (netarea, rooms, floor, year):
            return "", "Please fill required Tel Aviv fields."
        err = validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=floors_total)
        if err:
            return "", err
        try:
            X = build_tel_aviv_features(
                netarea, rooms, floor, year,
                gross_area=grossarea, floors=floors_total,
                apartments_in_building=apts,
                parking=parking, storage=storage, roof=roof, yard=yard,
                feature_cols_override=tel_aviv_v2_feature_cols,
            )
            pred_log = float(model_tel_aviv_v2.predict(X)[0])
            price = float(np.expm1(pred_log))
            price = max(price, 0.0)
            return f"Tel Aviv v2 — estimated price: {price:,.0f} ₪", ""
        except Exception as e:
            return "", f"Error during prediction: {e}"

    # Tel Aviv v1
    if model_choice == "tel_aviv_v1":
        if model_tel_aviv_v1 is None:
            return "", "Tel Aviv v1 model not available."
        if None in (netarea, rooms, floor, year):
            return "", "Please fill required Tel Aviv fields."
        err = validate_tel_aviv_inputs(netarea, rooms, floor, year)
        if err:
            return "", err
        try:
            X = np.array([[float(netarea), float(rooms), float(floor), float(year)]])
            price = float(model_tel_aviv_v1.predict(X)[0])
            return f"Tel Aviv v1 — estimated price: {price:,.0f} ₪", ""
        except Exception as e:
            return "", f"Error during prediction: {e}"

    # Taiwan
    if model_choice == "taiwan":
        if model_taiwan is None:
            return "", "Taiwan model not available."
        if None in (distance, convenience, lat, long_):
            return "", "Please fill all Taiwan fields."
        try:
            X = np.array([[float(distance), float(convenience), float(lat), float(long_)]])
            pred = float(model_taiwan.predict(X)[0])
            return f"Taiwan — predicted house price of unit area: {pred:,.2f}", ""
        except Exception as e:
            return "", f"Error during prediction: {e}"

    return "", "Selected model is not available."


if __name__ == "__main__":
    # Чтобы не плодились 2 процесса и не ловить “порт занят”
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DEBUG", "1") == "1"
    app.run(debug=debug, use_reloader=False, host="0.0.0.0", port=port)
