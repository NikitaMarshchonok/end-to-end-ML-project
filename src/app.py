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
# Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_TAIWAN_PATH = os.path.join(BASE_DIR, "models", "Real_estate_model.pkl")

# Старый Tel Aviv (v1) — 4 фичи
MODEL_TEL_AVIV_V1_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_real_estate_model.pkl")

# Новый Tel Aviv (v2) — 16+ фичей и log(price)
MODEL_TEL_AVIV_V2_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_real_estate_model_v2.pkl")
TEL_AVIV_V2_FEATS_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_feature_cols_v2.json")


# =========================================================
# Load models
# =========================================================

if not os.path.exists(MODEL_TAIWAN_PATH):
    raise FileNotFoundError(f"Taiwan model not found: {MODEL_TAIWAN_PATH}")

model_taiwan = joblib.load(MODEL_TAIWAN_PATH)

model_tel_aviv_v1 = None
if os.path.exists(MODEL_TEL_AVIV_V1_PATH):
    model_tel_aviv_v1 = joblib.load(MODEL_TEL_AVIV_V1_PATH)

model_tel_aviv_v2 = None
tel_aviv_v2_feature_cols = None
if os.path.exists(MODEL_TEL_AVIV_V2_PATH):
    model_tel_aviv_v2 = joblib.load(MODEL_TEL_AVIV_V2_PATH)
    if os.path.exists(TEL_AVIV_V2_FEATS_PATH):
        with open(TEL_AVIV_V2_FEATS_PATH, "r") as f:
            tel_aviv_v2_feature_cols = json.load(f)


# =========================================================
# Metrics / Model cards (hardcoded, portfolio-friendly)
# =========================================================

MODEL_CARDS = {
    "tel_aviv_v2": {
        "title": "Tel Aviv model v2 (RandomForest, log(price))",
        "mae": 669_657,
        "rmse": 1_418_990,
        "r2": 0.610,
        "note": "Trained on ~20 years of Tel Aviv transactions with engineered features + log(price).",
    },
    "tel_aviv_v1": {
        "title": "Tel Aviv model v1 (baseline)",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Baseline model with 4 raw inputs (net area, rooms, floor, year).",
    },
    "taiwan": {
        "title": "Tutorial model (Taiwan dataset)",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Classic tutorial dataset to demonstrate end-to-end ML + UI.",
    },
}


# =========================================================
# Fallback columns (если json не найден)
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
# Feature builder for Tel Aviv v2
# =========================================================

def build_tel_aviv_v2_features(
    netarea, rooms, floor, year,
    gross_area=None, floors=None,
    apartments_in_building=None,
    parking=None, storage=None, roof=None, yard=None
):
    """
    Build one-row feature vector for Tel Aviv v2 model.
    Model trained on log1p(price) => output needs expm1.
    """

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

    feature_cols = tel_aviv_v2_feature_cols or TEL_AVIV_V2_FALLBACK_COLS
    X = pd.DataFrame([row])

    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_cols]
    return X


# =========================================================
# Validation
# =========================================================

def validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=None):
    """
    Returns error string or None.
    """
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
# UI styles
# =========================================================

PAGE_STYLE = {
    "maxWidth": "760px",
    "margin": "36px auto",
    "padding": "0 14px",
    "fontFamily": "Inter, Arial, sans-serif",
}

HEADER_STYLE = {"textAlign": "center", "marginBottom": "6px"}
SUBHEADER_STYLE = {"textAlign": "center", "marginBottom": "24px", "color": "#555"}

CARD_STYLE = {
    "background": "#ffffff",
    "border": "1px solid #eee",
    "borderRadius": "10px",
    "padding": "18px 18px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.04)",
    "marginBottom": "14px",
}

MUTED_TEXT = {"fontSize": "12px", "color": "#666", "marginTop": "2px"}

LABEL_STYLE = {"fontWeight": "600", "fontSize": "13px", "marginBottom": "6px", "display": "block"}
INPUT_STYLE = {
    "width": "100%",
    "padding": "8px 10px",
    "borderRadius": "6px",
    "border": "1px solid #dcdcdc",
    "marginBottom": "12px",
}

BUTTON_STYLE = {
    "width": "100%",
    "padding": "12px 0",
    "backgroundColor": "#0d6efd",
    "color": "white",
    "border": "none",
    "borderRadius": "8px",
    "fontSize": "16px",
    "fontWeight": "600",
    "cursor": "pointer",
}

RESULT_CARD_STYLE = {
    **CARD_STYLE,
    "textAlign": "center",
    "fontSize": "18px",
    "fontWeight": "700",
    "minHeight": "50px",
}

ERROR_STYLE = {"color": "#b00020", "fontSize": "13px", "marginTop": "10px"}

MODEL_CARD_STYLE = {
    **CARD_STYLE,
    "background": "#f7f9ff",
    "borderLeft": "4px solid #0d6efd",
}


# =========================================================
# Dash app
# =========================================================

app = dash.Dash(__name__)
app.title = "Real Estate Price Prediction"

tel_aviv_options = []
if model_tel_aviv_v2 is not None:
    tel_aviv_options.append({"label": "Tel Aviv model v2 (improved, log(price))", "value": "tel_aviv_v2"})
if model_tel_aviv_v1 is not None:
    tel_aviv_options.append({"label": "Tel Aviv model v1 (baseline)", "value": "tel_aviv_v1"})

if not tel_aviv_options:
    tel_aviv_options = [{"label": "Tel Aviv model (not found)", "value": "tel_aviv_missing"}]

MODEL_OPTIONS = tel_aviv_options + [{"label": "Tutorial model (Taiwan dataset)", "value": "taiwan"}]


# =========================================================
# Presets for UX
# =========================================================

PRESETS = {
    "studio": {"netarea": 35, "rooms": 1.0, "floor": 2, "year": 1995},
    "family": {"netarea": 85, "rooms": 3.0, "floor": 5, "year": 2008},
    "penthouse": {"netarea": 150, "rooms": 4.5, "floor": 18, "year": 2020},
}


app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1("Real Estate Price Prediction", style=HEADER_STYLE),
        html.Div("Choose a model and fill the corresponding fields.", style=SUBHEADER_STYLE),

        # --- Model selector ---
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Label("Select model:", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="model-choice",
                    options=MODEL_OPTIONS,
                    value=MODEL_OPTIONS[0]["value"],
                    labelStyle={"display": "block", "marginTop": "6px"},
                    style={"fontSize": "14px"},
                ),
                html.Div(
                    "Tip: v2 usually gives better accuracy because it uses engineered features and log(price).",
                    style=MUTED_TEXT,
                ),
            ],
        ),

        # --- Model Card (dynamic) ---
        html.Div(id="model-card", style=MODEL_CARD_STYLE),

        # =====================================================
        # Tel Aviv required inputs + presets
        # =====================================================
        html.Div(
            id="section-telaviv",
            style=CARD_STYLE,
            children=[
                html.H3("Tel Aviv — required inputs", style={"marginTop": "0"}),

                html.Div(
                    children=[
                        html.Div("Presets:", style={**LABEL_STYLE, "marginBottom": "8px"}),
                        html.Div(
                            style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                            children=[
                                html.Button("1-room studio", id="preset-studio", n_clicks=0,
                                            style={"padding": "8px 10px", "borderRadius": "8px", "border": "1px solid #ddd", "cursor": "pointer"}),
                                html.Button("3-room family", id="preset-family", n_clicks=0,
                                            style={"padding": "8px 10px", "borderRadius": "8px", "border": "1px solid #ddd", "cursor": "pointer"}),
                                html.Button("Penthouse", id="preset-penthouse", n_clicks=0,
                                            style={"padding": "8px 10px", "borderRadius": "8px", "border": "1px solid #ddd", "cursor": "pointer"}),
                            ],
                        ),
                        html.Div("Tip: click a preset to auto-fill realistic values.", style=MUTED_TEXT),
                    ],
                    style={"marginBottom": "14px"},
                ),

                html.Label("Net area (m²) *", style=LABEL_STYLE),
                dcc.Input(id="input-netarea", type="number", placeholder="e.g. 80", style=INPUT_STYLE),

                html.Label("Number of rooms *", style=LABEL_STYLE),
                dcc.Input(id="input-rooms", type="number", placeholder="e.g. 3", style=INPUT_STYLE),

                html.Label("Floor *", style=LABEL_STYLE),
                dcc.Input(id="input-floor", type="number", placeholder="e.g. 4", style=INPUT_STYLE),

                html.Label("Construction year *", style=LABEL_STYLE),
                dcc.Input(id="input-year", type="number", placeholder="e.g. 2010", style={**INPUT_STYLE, "marginBottom": "0px"}),

                html.Div("(* required for Tel Aviv models v1/v2)", style=MUTED_TEXT),
            ],
        ),

        # =====================================================
        # Tel Aviv v2 optional fields
        # =====================================================
        html.Div(
            id="section-telaviv-v2-opt",
            style={**CARD_STYLE, "background": "#f7f9ff"},
            children=[
                html.H4("Tel Aviv v2 — optional fields", style={"marginTop": "0"}),
                html.Div("These improve accuracy for v2 if you know them. You can leave them empty.", style=MUTED_TEXT),

                html.Label("Gross area (m²)", style=LABEL_STYLE),
                dcc.Input(id="input-grossarea", type="number", placeholder="e.g. 95", style=INPUT_STYLE),

                html.Label("Total floors in building", style=LABEL_STYLE),
                dcc.Input(id="input-floors", type="number", placeholder="e.g. 12", style=INPUT_STYLE),

                html.Label("Apartments in building", style=LABEL_STYLE),
                dcc.Input(id="input-apts", type="number", placeholder="e.g. 40", style=INPUT_STYLE),

                html.Label("Parking spots", style=LABEL_STYLE),
                dcc.Input(id="input-parking", type="number", placeholder="e.g. 1", style=INPUT_STYLE),

                html.Label("Storage", style=LABEL_STYLE),
                dcc.Input(id="input-storage", type="number", placeholder="e.g. 1", style=INPUT_STYLE),

                html.Label("Roof area", style=LABEL_STYLE),
                dcc.Input(id="input-roof", type="number", placeholder="e.g. 20", style=INPUT_STYLE),

                html.Label("Yard area", style=LABEL_STYLE),
                dcc.Input(id="input-yard", type="number", placeholder="e.g. 30", style={**INPUT_STYLE, "marginBottom": "0px"}),
            ],
        ),

        # =====================================================
        # Taiwan inputs
        # =====================================================
        html.Div(
            id="section-taiwan",
            style=CARD_STYLE,
            children=[
                html.H3("Taiwan — inputs", style={"marginTop": "0"}),
                html.Div("Use these fields when 'Taiwan model' is selected above.", style=MUTED_TEXT),

                html.Label("Distance to the nearest MRT station", style=LABEL_STYLE),
                dcc.Input(id="input-distance", type="number", placeholder="e.g. 400", style=INPUT_STYLE),

                html.Label("Number of convenience stores", style=LABEL_STYLE),
                dcc.Input(id="input-convenience", type="number", placeholder="e.g. 4", style=INPUT_STYLE),

                html.Label("Latitude", style=LABEL_STYLE),
                dcc.Input(id="input-lat", type="number", placeholder="e.g. 24.98", style=INPUT_STYLE),

                html.Label("Longitude", style=LABEL_STYLE),
                dcc.Input(id="input-long", type="number", placeholder="e.g. 121.54", style={**INPUT_STYLE, "marginBottom": "0px"}),
            ],
        ),

        # =====================================================
        # Action
        # =====================================================
        html.Div(
            style={**CARD_STYLE, "padding": "14px 18px"},
            children=[
                html.Button("Predict price", id="predict-button", n_clicks=0, style=BUTTON_STYLE),
                html.Div(id="validation-output", style=ERROR_STYLE),
            ],
        ),

        # =====================================================
        # Result
        # =====================================================
        html.Div(id="prediction-output", style=RESULT_CARD_STYLE, children=""),
    ],
)


# =========================================================
# UI visibility callback
# =========================================================

@app.callback(
    Output("section-telaviv", "style"),
    Output("section-telaviv-v2-opt", "style"),
    Output("section-taiwan", "style"),
    Input("model-choice", "value"),
)
def toggle_sections(model_choice):
    show_telaviv = model_choice in ("tel_aviv_v1", "tel_aviv_v2")
    show_telaviv_v2_opt = model_choice == "tel_aviv_v2"
    show_taiwan = model_choice == "taiwan"

    telaviv_style = {**CARD_STYLE, "display": "block"} if show_telaviv else {**CARD_STYLE, "display": "none"}
    telaviv_v2_style = {**CARD_STYLE, "background": "#f7f9ff", "display": "block"} if show_telaviv_v2_opt else {**CARD_STYLE, "display": "none"}
    taiwan_style = {**CARD_STYLE, "display": "block"} if show_taiwan else {**CARD_STYLE, "display": "none"}

    return telaviv_style, telaviv_v2_style, taiwan_style


# =========================================================
# Button label callback
# =========================================================

@app.callback(
    Output("predict-button", "children"),
    Input("model-choice", "value"),
)
def update_button_text(model_choice):
    if model_choice == "tel_aviv_v2":
        return "Predict with Tel Aviv v2"
    if model_choice == "tel_aviv_v1":
        return "Predict with Tel Aviv v1"
    if model_choice == "taiwan":
        return "Predict with Taiwan model"
    return "Predict price"


# =========================================================
# Model Card callback
# =========================================================

@app.callback(
    Output("model-card", "children"),
    Input("model-choice", "value"),
)
def update_model_card(model_choice):
    card = MODEL_CARDS.get(model_choice)
    if not card:
        return [
            html.H4("Model card", style={"marginTop": "0"}),
            html.Div("No model info available.", style=MUTED_TEXT),
        ]

    def fmt_money(v):
        return f"{int(v):,} ₪" if v is not None else "—"

    def fmt_num(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    return [
        html.H4("Model card", style={"marginTop": "0"}),
        html.Div(card["title"], style={"fontWeight": "700", "marginBottom": "8px"}),
        html.Ul(
            style={"marginTop": "0", "marginBottom": "8px"},
            children=[
                html.Li(f"MAE ≈ {fmt_money(card.get('mae'))}"),
                html.Li(f"RMSE ≈ {fmt_money(card.get('rmse'))}"),
                html.Li(f"R² ≈ {fmt_num(card.get('r2'))}"),
            ],
        ),
        html.Div(card.get("note", ""), style=MUTED_TEXT),
    ]


# =========================================================
# Presets callback
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
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "preset-studio":
        p = PRESETS["studio"]
    elif trig == "preset-family":
        p = PRESETS["family"]
    else:
        p = PRESETS["penthouse"]

    return p["netarea"], p["rooms"], p["floor"], p["year"]


# =========================================================
# Prediction callback (with validation)
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

    # Tel Aviv v2 optional
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
    n_clicks,
    model_choice,
    netarea, rooms, floor, year,
    grossarea, floors_total, apts, parking, storage, roof, yard,
    distance, convenience, lat, long_,
):
    if not n_clicks:
        return "", ""

    # -------------------------
    # Tel Aviv v2
    # -------------------------
    if model_choice == "tel_aviv_v2":
        if netarea is None or rooms is None or floor is None or year is None:
            return "", "Please fill in all required Tel Aviv fields (net area, rooms, floor, year)."

        err = validate_tel_aviv_inputs(netarea, rooms, floor, year, floors_total=floors_total)
        if err:
            return "", err

        if model_tel_aviv_v2 is None:
            return "", "Tel Aviv v2 model file not found."

        try:
            X = build_tel_aviv_v2_features(
                netarea=netarea,
                rooms=rooms,
                floor=floor,
                year=year,
                gross_area=grossarea,
                floors=floors_total,
                apartments_in_building=apts,
                parking=parking,
                storage=storage,
                roof=roof,
                yard=yard,
            )
            pred_log = model_tel_aviv_v2.predict(X)[0]
            predicted_price = float(np.expm1(pred_log))
            return f"Tel Aviv model v2 — estimated apartment price: {predicted_price:,.0f} ₪", ""
        except Exception as e:
            return "", f"Error during Tel Aviv v2 prediction: {e}"

    # -------------------------
    # Tel Aviv v1
    # -------------------------
    if model_choice == "tel_aviv_v1":
        if netarea is None or rooms is None or floor is None or year is None:
            return "", "Please fill in all required Tel Aviv fields (net area, rooms, floor, year)."

        err = validate_tel_aviv_inputs(netarea, rooms, floor, year)
        if err:
            return "", err

        if model_tel_aviv_v1 is None:
            return "", "Tel Aviv v1 model file not found."

        try:
            features = np.array([[float(netarea), float(rooms), float(floor), float(year)]])
            predicted_price = model_tel_aviv_v1.predict(features)[0]
            return f"Tel Aviv model v1 — estimated apartment price: {predicted_price:,.0f} ₪", ""
        except Exception as e:
            return "", f"Error during Tel Aviv v1 prediction: {e}"

    # -------------------------
    # Taiwan
    # -------------------------
    if model_choice == "taiwan":
        if distance is None or convenience is None or lat is None or long_ is None:
            return "", "Please fill in all Taiwan fields."

        try:
            features = np.array([[float(distance), float(convenience), float(lat), float(long_)]])
            predicted_price = model_taiwan.predict(features)[0]
            return f"Taiwan model — predicted house price of unit area: {predicted_price:,.2f}", ""
        except Exception as e:
            return "", f"Error during Taiwan prediction: {e}"

    return "", "Selected model is not available."


if __name__ == "__main__":
    app.run(debug=True)
