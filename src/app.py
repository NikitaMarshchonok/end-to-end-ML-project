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


# Tel Aviv v1 (optional but desirable)
model_tel_aviv_v1 = None
if os.path.exists(MODEL_TEL_AVIV_V1_PATH):
    model_tel_aviv_v1 = joblib.load(MODEL_TEL_AVIV_V1_PATH)

# Tel Aviv v2 (preferred)
model_tel_aviv_v2 = None
tel_aviv_v2_feature_cols = None

if os.path.exists(MODEL_TEL_AVIV_V2_PATH):
    model_tel_aviv_v2 = joblib.load(MODEL_TEL_AVIV_V2_PATH)

    if os.path.exists(TEL_AVIV_V2_FEATS_PATH):
        with open(TEL_AVIV_V2_FEATS_PATH, "r") as f:
            tel_aviv_v2_feature_cols = json.load(f)


# Fallback list (если json не найден)
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

def build_tel_aviv_v2_features(netarea, rooms, floor, year,
                               gross_area=None, floors=None,
                               apartments_in_building=None,
                               parking=None, storage=None, roof=None, yard=None):
    """
    Собираем одну строку фичей для улучшенной модели Tel Aviv v2.
    Модель обучалась на log1p(price), поэтому на выходе нужно expm1.
    """

    # базовые
    netarea = float(netarea)
    rooms = float(rooms)
    floor = float(floor)
    year = float(year)

    # опциональные поля (в UI их пока нет — оставим NaN)
    gross_area = float(gross_area) if gross_area not in [None, ""] else np.nan
    floors = float(floors) if floors not in [None, ""] else np.nan
    apartments_in_building = float(apartments_in_building) if apartments_in_building not in [None, ""] else np.nan

    parking = float(parking) if parking not in [None, ""] else np.nan
    storage = float(storage) if storage not in [None, ""] else np.nan
    roof = float(roof) if roof not in [None, ""] else np.nan
    yard = float(yard) if yard not in [None, ""] else np.nan

    # безопасный grossArea
    if np.isnan(gross_area):
        gross_area = netarea

    # временные признаки - используем текущую дату как "proxy"
    now = datetime.now()
    tx_year = now.year
    tx_month = now.month
    tx_quarter = (tx_month - 1) // 3 + 1

    # возраст здания на момент сделки
    building_age_at_tx = np.nan
    if year and year > 0:
        building_age_at_tx = max(0, tx_year - int(year))

    # floor_ratio
    floor_ratio = np.nan
    if floors and not np.isnan(floors) and floors > 0:
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

    # гарантируем полный набор колонок
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_cols]
    return X


# =========================================================
# Dash app
# =========================================================

app = dash.Dash(__name__)
app.title = "Real Estate Price Prediction"

# определяем, какие Tel Aviv варианты доступны
tel_aviv_options = []
if model_tel_aviv_v2 is not None:
    tel_aviv_options.append({"label": "Tel Aviv model v2 (improved, log(price))", "value": "tel_aviv_v2"})
if model_tel_aviv_v1 is not None:
    tel_aviv_options.append({"label": "Tel Aviv model v1 (baseline)", "value": "tel_aviv_v1"})

# если вдруг нет ни одной Tel Aviv модели — всё равно оставим пункт-заглушку
if not tel_aviv_options:
    tel_aviv_options = [{"label": "Tel Aviv model (not found)", "value": "tel_aviv_missing"}]

app.layout = html.Div(
    style={
        "maxWidth": "700px",
        "margin": "40px auto",
        "fontFamily": "Arial, sans-serif",
    },
    children=[
        html.H1(
            "Real Estate Price Prediction",
            style={"textAlign": "center", "marginBottom": "10px"},
        ),

        html.P(
            "Choose a model and fill the corresponding fields.",
            style={"textAlign": "center", "marginBottom": "25px"},
        ),

        # --- Model selector ---
        html.Div(
            style={"marginBottom": "30px"},
            children=[
                html.Label("Select model:"),
                dcc.RadioItems(
                    id="model-choice",
                    options=(
                        tel_aviv_options
                        + [
                            {"label": "Tutorial model (Taiwan dataset)", "value": "taiwan"},
                        ]
                    ),
                    value=tel_aviv_options[0]["value"],
                    labelStyle={"display": "block", "marginTop": "5px"},
                ),
            ],
        ),

        html.Hr(),

        # --- Tel Aviv inputs (минимальный набор) ---
        html.H3("Tel Aviv model — inputs", style={"marginTop": "20px"}),
        html.P("Use these fields when 'Tel Aviv model' is selected above."),

        html.Label("Net area (m²)"),
        dcc.Input(
            id="input-netarea",
            type="number",
            placeholder="e.g. 80",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Number of rooms"),
        dcc.Input(
            id="input-rooms",
            type="number",
            placeholder="e.g. 3",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Floor"),
        dcc.Input(
            id="input-floor",
            type="number",
            placeholder="e.g. 4",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Construction year"),
        dcc.Input(
            id="input-year",
            type="number",
            placeholder="e.g. 2010",
            style={"width": "100%", "marginBottom": "20px"},
        ),

        html.Hr(style={"margin": "25px 0"}),

        # --- Taiwan inputs ---
        html.H3("Taiwan model — inputs", style={"marginTop": "10px"}),
        html.P("Use these fields when 'Taiwan model' is selected above."),

        html.Label("Distance to the nearest MRT station"),
        dcc.Input(
            id="input-distance",
            type="number",
            placeholder="e.g. 400",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Number of convenience stores"),
        dcc.Input(
            id="input-convenience",
            type="number",
            placeholder="e.g. 4",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Latitude"),
        dcc.Input(
            id="input-lat",
            type="number",
            placeholder="e.g. 24.98",
            style={"width": "100%", "marginBottom": "10px"},
        ),

        html.Label("Longitude"),
        dcc.Input(
            id="input-long",
            type="number",
            placeholder="e.g. 121.54",
            style={"width": "100%", "marginBottom": "20px"},
        ),

        html.Button(
            "Predict price",
            id="predict-button",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "10px 0",
                "backgroundColor": "#007bff",
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "fontSize": "16px",
                "cursor": "pointer",
            },
        ),

        html.Hr(style={"margin": "30px 0"}),

        html.Div(
            id="prediction-output",
            style={
                "textAlign": "center",
                "fontSize": "20px",
                "fontWeight": "bold",
                "minHeight": "30px",
            },
        ),
    ],
)


# =========================================================
# Callback
# =========================================================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("model-choice", "value"),
    State("input-netarea", "value"),
    State("input-rooms", "value"),
    State("input-floor", "value"),
    State("input-year", "value"),
    State("input-distance", "value"),
    State("input-convenience", "value"),
    State("input-lat", "value"),
    State("input-long", "value"),
)
def predict_price(
    n_clicks,
    model_choice,
    netarea,
    rooms,
    floor,
    year,
    distance,
    convenience,
    lat,
    long_,
):
    if not n_clicks:
        return ""

    # -------------------------
    # Tel Aviv v2
    # -------------------------
    if model_choice == "tel_aviv_v2":
        if netarea is None or rooms is None or floor is None or year is None:
            return "Please fill in all Tel Aviv fields."

        if model_tel_aviv_v2 is None:
            return "Tel Aviv v2 model file not found."

        try:
            X = build_tel_aviv_v2_features(
                netarea=netarea,
                rooms=rooms,
                floor=floor,
                year=year,
            )
            pred_log = model_tel_aviv_v2.predict(X)[0]
            predicted_price = float(np.expm1(pred_log))
            return f"Tel Aviv model v2 — estimated apartment price: {predicted_price:,.0f} ₪"
        except Exception as e:
            return f"Error during Tel Aviv v2 prediction: {e}"

    # -------------------------
    # Tel Aviv v1 (старый)
    # -------------------------
    if model_choice == "tel_aviv_v1":
        if netarea is None or rooms is None or floor is None or year is None:
            return "Please fill in all Tel Aviv fields."

        if model_tel_aviv_v1 is None:
            return "Tel Aviv v1 model file not found."

        try:
            features = np.array([[float(netarea), float(rooms), float(floor), float(year)]])
            predicted_price = model_tel_aviv_v1.predict(features)[0]
            return f"Tel Aviv model v1 — estimated apartment price: {predicted_price:,.0f} ₪"
        except Exception as e:
            return f"Error during Tel Aviv v1 prediction: {e}"

    # -------------------------
    # Taiwan
    # -------------------------
    if model_choice == "taiwan":
        if distance is None or convenience is None or lat is None or long_ is None:
            return "Please fill in all Taiwan fields."

        try:
            features = np.array([[float(distance), float(convenience), float(lat), float(long_)]])
            predicted_price = model_taiwan.predict(features)[0]
            return "Taiwan model — predicted house price of unit area: " f"{predicted_price:,.2f}"
        except Exception as e:
            return f"Error during Taiwan prediction: {e}"

    # -------------------------
    # Fallback
    # -------------------------
    return "Selected model is not available."


if __name__ == "__main__":
    app.run(debug=True)
