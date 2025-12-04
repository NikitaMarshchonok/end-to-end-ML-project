# src/app.py

from pathlib import Path

import joblib
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State

# === Пути ===
# PROJECT_ROOT = .../end-to-end Ml project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "real_estate_model.pkl"

# === Загружаем обученную модель один раз при старте приложения ===
model = joblib.load(MODEL_PATH)

# ВАЖНО: имена признаков должны совпадать с тем, как мы обучали модель в train_model.py
FEATURE_COLUMNS = [
    "Distance to the nearest MRT station",
    "Number of convenience stores",
    "Latitude",
    "Longitude",
]

# === Создаём Dash-приложение ===
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Real Estate Price Prediction",
            style={"textAlign": "center", "marginBottom": "30px"},
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Distance to the nearest MRT station"),
                        dcc.Input(
                            id="distance_to_mrt",
                            type="number",
                            placeholder="e.g. 500",
                            style={"width": "100%", "padding": "8px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.Label("Number of convenience stores"),
                        dcc.Input(
                            id="num_convenience_stores",
                            type="number",
                            placeholder="e.g. 5",
                            style={"width": "100%", "padding": "8px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.Label("Latitude"),
                        dcc.Input(
                            id="latitude",
                            type="number",
                            placeholder="e.g. 24.96",
                            style={"width": "100%", "padding": "8px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.Label("Longitude"),
                        dcc.Input(
                            id="longitude",
                            type="number",
                            placeholder="e.g. 121.54",
                            style={"width": "100%", "padding": "8px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),

                html.Button(
                    "Predict Price",
                    id="predict_button",
                    n_clicks=0,
                    style={
                        "marginTop": "10px",
                        "padding": "10px 20px",
                        "fontSize": "16px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                "maxWidth": "400px",
                "margin": "0 auto",
                "display": "flex",
                "flexDirection": "column",
            },
        ),

        html.Hr(),

        html.Div(
            id="prediction_output",
            style={"textAlign": "center", "fontSize": "20px", "marginTop": "20px"},
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "40px",
    },
)


# === Callback: логика предсказания ===
@app.callback(
    Output("prediction_output", "children"),
    Input("predict_button", "n_clicks"),
    State("distance_to_mrt", "value"),
    State("num_convenience_stores", "value"),
    State("latitude", "value"),
    State("longitude", "value"),
)
def update_prediction(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    """Вызывается при нажатии кнопки Predict Price."""
    if n_clicks is None or n_clicks == 0:
        # Кнопку ещё не нажимали — ничего не показываем
        return ""

    # Проверяем, что пользователь заполнил все поля
    if None in (distance_to_mrt, num_convenience_stores, latitude, longitude):
        return "Please enter all values to get a prediction."

    # Формируем DataFrame с теми же колонками, что при обучении
    features_df = pd.DataFrame(
        [[distance_to_mrt, num_convenience_stores, latitude, longitude]],
        columns=FEATURE_COLUMNS,
    )

    # Делаем предсказание
    predicted_price = model.predict(features_df)[0]

    return f"Predicted House Price of Unit Area: {predicted_price:.2f}"


if __name__ == "__main__":
    app.run(debug=True)
