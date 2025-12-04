import os
import joblib
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# === Пути к моделям ===

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_TAIWAN_PATH = os.path.join(BASE_DIR, "models", "Real_estate_model.pkl")
MODEL_TEL_AVIV_PATH = os.path.join(BASE_DIR, "models", "tel_aviv_real_estate_model.pkl")

if not os.path.exists(MODEL_TAIWAN_PATH):
    raise FileNotFoundError(f"Taiwan model not found: {MODEL_TAIWAN_PATH}")

if not os.path.exists(MODEL_TEL_AVIV_PATH):
    raise FileNotFoundError(f"Tel Aviv model not found: {MODEL_TEL_AVIV_PATH}")

model_taiwan = joblib.load(MODEL_TAIWAN_PATH)
model_tel_aviv = joblib.load(MODEL_TEL_AVIV_PATH)

# === Создаём Dash-приложение ===

app = dash.Dash(__name__)
app.title = "Real Estate Price Prediction"

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

        # --- Переключатель модели ---
        html.Div(
            style={"marginBottom": "30px"},
            children=[
                html.Label("Select model:"),
                dcc.RadioItems(
                    id="model-choice",
                    options=[
                        {
                            "label": "Tel Aviv model (Israel dataset)",
                            "value": "tel_aviv",
                        },
                        {
                            "label": "Tutorial model (Taiwan dataset)",
                            "value": "taiwan",
                        },
                    ],
                    value="tel_aviv",
                    labelStyle={"display": "block", "marginTop": "5px"},
                ),
            ],
        ),

        html.Hr(),

        # --- Блок для модели Тель-Авива ---
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

        # --- Блок для модели Taiwan (первая статья) ---
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

# === Коллбэк предсказания ===

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

    # --- Ветвь для Тель-Авива ---
    if model_choice == "tel_aviv":
        if netarea is None or rooms is None or floor is None or year is None:
            return "Please fill in all Tel Aviv fields."

        try:
            features = np.array(
                [[float(netarea), float(rooms), float(floor), float(year)]]
            )
            predicted_price = model_tel_aviv.predict(features)[0]
            return f"Tel Aviv model — estimated apartment price: {predicted_price:,.0f} ₪"
        except Exception as e:
            return f"Error during Tel Aviv prediction: {e}"

    # --- Ветвь для Taiwan ---
    else:
        if (
            distance is None
            or convenience is None
            or lat is None
            or long_ is None
        ):
            return "Please fill in all Taiwan fields."

        try:
            # порядок признаков: [distance_to_MRT, number_of_convenience_stores, latitude, longitude]
            features = np.array(
                [
                    [
                        float(distance),
                        float(convenience),
                        float(lat),
                        float(long_),
                    ]
                ]
            )
            predicted_price = model_taiwan.predict(features)[0]
            return (
                "Taiwan model — predicted house price of unit area: "
                f"{predicted_price:,.2f}"
            )
        except Exception as e:
            return f"Error during Taiwan prediction: {e}"


if __name__ == "__main__":
    # Для новой версии Dash используем run()
    app.run(debug=True)
