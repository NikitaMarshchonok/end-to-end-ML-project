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


# =========================================================
# Model metrics for "Model Card"
# (значения взяты из ноутбука с экспериментами)
# =========================================================

MODEL_METRICS = {
    "tel_aviv_v2": {
        "display_name": "Tel Aviv model v2 (RandomForest, log(price))",
        "mae": 669_657,
        "rmse": 1_418_990,
        "r2": 0.610,
        "note": "Trained on 20 years of Tel Aviv transactions with engineered features and log(price).",
    },
    "tel_aviv_v1": {
        "display_name": "Tel Aviv model v1 (Linear regression, no feature engineering)",
        "mae": 858_171,
        "rmse": 1_778_791,
        "r2": 0.387,
        "note": "Baseline linear model with a smaller feature set.",
    },
    "taiwan": {
        "display_name": "Tutorial model (Taiwan housing dataset)",
        "mae": None,
        "rmse": None,
        "r2": None,
        "note": "Educational model trained on the classic Taiwan real estate dataset.",
    },
}


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

def build_tel_aviv_v2_features(
    netarea, rooms, floor, year,
    gross_area=None, floors=None,
    apartments_in_building=None,
    parking=None, storage=None, roof=None, yard=None
):
    """
    Собираем одну строку фичей для улучшенной модели Tel Aviv v2.
    Модель обучалась на log1p(price), поэтому на выходе нужно expm1.
    """

    # базовые
    netarea = float(netarea)
    rooms = float(rooms)
    floor = float(floor)
    year = float(year)

    # опциональные поля
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

    # гарантируем полный набор колонок
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan

    X = X[feature_cols]
    return X


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
    "fontSize": "20px",
    "fontWeight": "700",
    "minHeight": "50px",
}


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

MODEL_OPTIONS = (
    tel_aviv_options
    + [{"label": "Tutorial model (Taiwan dataset)", "value": "taiwan"}]
)


app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1("Real Estate Price Prediction", style=HEADER_STYLE),
        html.Div(
            "Choose a model and fill the corresponding fields.",
            style=SUBHEADER_STYLE,
        ),

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

        # --- Model Card (метрики модели) ---
        html.Div(
            id="model-card",
            style={
                **CARD_STYLE,
                "borderLeft": "4px solid #0d6efd",
                "background": "#f8f9ff",
                "marginTop": "-4px",
            },
        ),

        # =====================================================
        # Tel Aviv базовые поля
        # =====================================================
        html.Div(
            id="section-telaviv",
            style=CARD_STYLE,
            children=[
                html.H3("Tel Aviv — required inputs", style={"marginTop": "0"}),

                html.Label("Net area (m²) *", style=LABEL_STYLE),
                dcc.Input(
                    id="input-netarea",
                    type="number",
                    placeholder="e.g. 80",
                    style=INPUT_STYLE,
                ),

                html.Label("Number of rooms *", style=LABEL_STYLE),
                dcc.Input(
                    id="input-rooms",
                    type="number",
                    placeholder="e.g. 3",
                    style=INPUT_STYLE,
                ),

                html.Label("Floor *", style=LABEL_STYLE),
                dcc.Input(
                    id="input-floor",
                    type="number",
                    placeholder="e.g. 4",
                    style=INPUT_STYLE,
                ),

                html.Label("Construction year *", style=LABEL_STYLE),
                dcc.Input(
                    id="input-year",
                    type="number",
                    placeholder="e.g. 2010",
                    style={**INPUT_STYLE, "marginBottom": "0px"},
                ),

                html.Div(
                    "(* required for Tel Aviv models v1/v2)",
                    style=MUTED_TEXT,
                ),
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
                html.Div(
                    "These improve accuracy for v2 if you know them. You can leave them empty.",
                    style=MUTED_TEXT,
                ),

                html.Label("Gross area (m²)", style=LABEL_STYLE),
                dcc.Input(
                    id="input-grossarea",
                    type="number",
                    placeholder="e.g. 95",
                    style=INPUT_STYLE,
                ),

                html.Label("Total floors in building", style=LABEL_STYLE),
                dcc.Input(
                    id="input-floors",
                    type="number",
                    placeholder="e.g. 12",
                    style=INPUT_STYLE,
                ),

                html.Label("Apartments in building", style=LABEL_STYLE),
                dcc.Input(
                    id="input-apts",
                    type="number",
                    placeholder="e.g. 40",
                    style=INPUT_STYLE,
                ),

                html.Label("Parking spots", style=LABEL_STYLE),
                dcc.Input(
                    id="input-parking",
                    type="number",
                    placeholder="e.g. 1",
                    style=INPUT_STYLE,
                ),

                html.Label("Storage", style=LABEL_STYLE),
                dcc.Input(
                    id="input-storage",
                    type="number",
                    placeholder="e.g. 1",
                    style=INPUT_STYLE,
                ),

                html.Label("Roof area", style=LABEL_STYLE),
                dcc.Input(
                    id="input-roof",
                    type="number",
                    placeholder="e.g. 20",
                    style=INPUT_STYLE,
                ),

                html.Label("Yard area", style=LABEL_STYLE),
                dcc.Input(
                    id="input-yard",
                    type="number",
                    placeholder="e.g. 30",
                    style={**INPUT_STYLE, "marginBottom": "0px"},
                ),
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
                dcc.Input(
                    id="input-distance",
                    type="number",
                    placeholder="e.g. 400",
                    style=INPUT_STYLE,
                ),

                html.Label("Number of convenience stores", style=LABEL_STYLE),
                dcc.Input(
                    id="input-convenience",
                    type="number",
                    placeholder="e.g. 4",
                    style=INPUT_STYLE,
                ),

                html.Label("Latitude", style=LABEL_STYLE),
                dcc.Input(
                    id="input-lat",
                    type="number",
                    placeholder="e.g. 24.98",
                    style=INPUT_STYLE,
                ),

                html.Label("Longitude", style=LABEL_STYLE),
                dcc.Input(
                    id="input-long",
                    type="number",
                    placeholder="e.g. 121.54",
                    style={**INPUT_STYLE, "marginBottom": "0px"},
                ),
            ],
        ),

        # =====================================================
        # Action
        # =====================================================
        html.Div(
            style={**CARD_STYLE, "padding": "14px 18px"},
            children=[
                html.Button(
                    "Predict price",
                    id="predict-button",
                    n_clicks=0,
                    style=BUTTON_STYLE,
                ),
            ],
        ),

        # =====================================================
        # Result
        # =====================================================
        html.Div(
            id="prediction-output",
            style=RESULT_CARD_STYLE,
            children="",
        ),
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
    telaviv_v2_style = {
        **CARD_STYLE,
        "background": "#f7f9ff",
        "display": "block",
    } if show_telaviv_v2_opt else {**CARD_STYLE, "display": "none"}
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
# Model card callback
# =========================================================

@app.callback(
    Output("model-card", "children"),
    Input("model-choice", "value"),
)
def update_model_card(model_choice: str):
    info = MODEL_METRICS.get(model_choice)

    if not info:
        return html.Div(
            "No metrics available for this model yet.",
            style={"fontSize": "13px"},
        )

    mae = info.get("mae")
    rmse = info.get("rmse")
    r2 = info.get("r2")

    metrics_block = []
    if mae is not None:
        metrics_block.append(html.Li(f"MAE  ≈ {mae:,.0f} ₪"))
    if rmse is not None:
        metrics_block.append(html.Li(f"RMSE ≈ {rmse:,.0f} ₪"))
    if r2 is not None:
        metrics_block.append(html.Li(f"R²   ≈ {r2:.3f}"))

    return html.Div(
        children=[
            html.Div("Model card", style={"fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"}),
            html.Div(info["display_name"], style={"fontSize": "13px", "marginBottom": "6px"}),
            html.Ul(metrics_block, style={"paddingLeft": "18px", "margin": "0 0 4px 0", "fontSize": "13px"}),
            html.Div(
                info.get("note", ""),
                style={**MUTED_TEXT, "marginTop": "4px"},
            ),
        ]
    )


# =========================================================
# Validation helpers
# =========================================================

def validate_tel_aviv_inputs(netarea, rooms, floor, year):
    """
    Простая валидация основных полей для Tel Aviv.
    Возвращает список строк-ошибок (если он пустой — всё ок).
    """
    errors = []
    current_year = datetime.now().year

    # Net area
    if netarea is None:
        errors.append("Please enter net area (m²).")
    else:
        try:
            na = float(netarea)
            if na <= 10 or na > 400:
                errors.append("Net area should be between 10 and 400 m².")
        except ValueError:
            errors.append("Net area must be a number.")

    # Rooms
    if rooms is None:
        errors.append("Please enter number of rooms.")
    else:
        try:
            r = float(rooms)
            if r <= 0 or r > 10:
                errors.append("Number of rooms should be between 1 and 10.")
        except ValueError:
            errors.append("Number of rooms must be a number.")

    # Floor
    if floor is None:
        errors.append("Please enter floor.")
    else:
        try:
            f = float(floor)
            if f < 0 or f > 60:
                errors.append("Floor should be between 0 and 60.")
        except ValueError:
            errors.append("Floor must be a number.")

    # Year
    if year is None:
        errors.append("Please enter construction year.")
    else:
        try:
            y = int(year)
            if y < 1960 or y > current_year:
                errors.append(f"Construction year should be between 1960 and {current_year}.")
        except ValueError:
            errors.append("Construction year must be an integer.")

    return errors


# =========================================================
# Prediction callback
# =========================================================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("model-choice", "value"),
    State("input-netarea", "value"),
    State("input-rooms", "value"),
    State("input-floor", "value"),
    State("input-year", "value"),

    # --- Tel Aviv v2 optional states ---
    State("input-grossarea", "value"),
    State("input-floors", "value"),
    State("input-apts", "value"),
    State("input-parking", "value"),
    State("input-storage", "value"),
    State("input-roof", "value"),
    State("input-yard", "value"),

    # --- Taiwan states ---
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

    grossarea,
    floors,
    apts,
    parking,
    storage,
    roof,
    yard,

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
        # Быстрая проверка на заполненность
        if netarea is None or rooms is None or floor is None or year is None:
            return "Please fill in all required Tel Aviv fields (net area, rooms, floor, year)."

        # Дополнительная валидация диапазонов
        errors = validate_tel_aviv_inputs(netarea, rooms, floor, year)
        if errors:
            return "Input validation error: " + " ".join(errors)

        if model_tel_aviv_v2 is None:
            return "Tel Aviv v2 model file not found."

        try:
            X = build_tel_aviv_v2_features(
                netarea=netarea,
                rooms=rooms,
                floor=floor,
                year=year,
                gross_area=grossarea,
                floors=floors,
                apartments_in_building=apts,
                parking=parking,
                storage=storage,
                roof=roof,
                yard=yard,
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
            return "Please fill in all required Tel Aviv fields (net area, rooms, floor, year)."

        errors = validate_tel_aviv_inputs(netarea, rooms, floor, year)
        if errors:
            return "Input validation error: " + " ".join(errors)

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
