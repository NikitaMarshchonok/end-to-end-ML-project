import os

import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from ml.predictor import (
    TEL_AVIV_V2_FALLBACK_COLS,
    get_feature_importances,
    get_model_spec,
    predict,
)

# =========================================================
# Build tag (чтобы ты видел, что запустился ИМЕННО этот файл)
# =========================================================
APP_BUILD = "UI_v4_uncertainty_quantiles__2025-12-24"


# =========================================================
# UI helpers
# =========================================================
def ui_result(title: str, value: str, subtitle=None):
    children = [
        html.Div(title.upper(), className="resultTitle"),
        html.Div(value, className="resultValue"),
    ]

    if subtitle:
        if isinstance(subtitle, (list, tuple)):
            for s in subtitle:
                if s is None:
                    continue
                # dash components can be appended directly
                if hasattr(s, "to_plotly_json"):
                    children.append(s)
                else:
                    children.append(html.Div(str(s), className="resultSub"))
        else:
            if hasattr(subtitle, "to_plotly_json"):
                children.append(subtitle)
            else:
                children.append(html.Div(str(subtitle), className="resultSub"))

    return html.Div(children, className="resultInner")


def build_feature_importance_figure(model_obj, feature_cols: list[str], top_k: int = 12):
    imp = get_feature_importances(model_obj)
    if imp is None or feature_cols is None:
        fig = go.Figure()
        fig.update_layout(
            height=120,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="Feature importance is not available for this model.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=13),
                )
            ],
        )
        return fig

    imp = np.array(list(imp), dtype=float)
    n = min(len(feature_cols), len(imp))
    feat = np.array(feature_cols[:n], dtype=object)
    imp = imp[:n]

    order = np.argsort(imp)[-top_k:]
    feat_top = feat[order]
    imp_top = imp[order]

    fig = go.Figure(data=[go.Bar(x=imp_top, y=feat_top, orientation="h")])
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="importance", gridcolor="rgba(148,163,184,.25)", zeroline=False),
        yaxis=dict(title="", automargin=True, gridcolor="rgba(148,163,184,.10)"),
        showlegend=False,
    )
    return fig


# =========================================================
# Load models from shared predictor
# =========================================================
spec_taiwan = get_model_spec("taiwan")
spec_tel_aviv_v1 = get_model_spec("tel_aviv_v1")
spec_tel_aviv_v2 = get_model_spec("tel_aviv_v2")
spec_tel_aviv_v3_2 = get_model_spec("tel_aviv_v3_2_clean")

model_taiwan = spec_taiwan.model if spec_taiwan else None
model_tel_aviv_v1 = spec_tel_aviv_v1.model if spec_tel_aviv_v1 else None
model_tel_aviv_v2 = spec_tel_aviv_v2.model if spec_tel_aviv_v2 else None
model_tel_aviv_v3_2 = spec_tel_aviv_v3_2.model if spec_tel_aviv_v3_2 else None

tel_aviv_v2_feature_cols = spec_tel_aviv_v2.feature_cols if spec_tel_aviv_v2 else None
tel_aviv_v3_2_feature_cols = spec_tel_aviv_v3_2.feature_cols if spec_tel_aviv_v3_2 else None
tel_aviv_v3_2_metrics = spec_tel_aviv_v3_2.metrics if spec_tel_aviv_v3_2 else None


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
# Prediction range helpers
# =========================================================
def fmt_money(v: float, currency: str) -> str:
    if currency == "ILS":
        return f"{v:,.0f} ₪"
    return f"{v:,.2f} {currency}"


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

/* Alerts + results */
.alert{
  margin-top:12px;
  padding:12px 14px;
  border-radius:14px;
  border:1px solid var(--border);
  font-weight:800;
}
.alert:empty{display:none}
.alertError{background:#fff1f2;border-color:#fecdd3;color:#9f1239}
.result:empty{display:none}
.resultInner{display:flex;flex-direction:column;gap:6px;align-items:center}
.resultLabel{font-size:12px;color:var(--muted);font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.resultValue{font-size:28px;font-weight:1000;letter-spacing:-.02em}
.resultSub{font-size:12px;color:var(--muted);font-weight:700}

/* Graph */
.graphWrap{margin-top:10px}

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

        # Explainability / feature importance (RF models)
        html.Div(
            id="section-fi",
            className="card",
            children=[
                html.Div("Explainability", className="cardTitle"),
                html.Div("Top features driving the prediction (RandomForest).", className="muted"),
                html.Div(className="hr"),
                html.Div(
                    className="graphWrap",
                    children=[
                        dcc.Graph(
                            id="feat-importance-graph",
                            config={"displayModeBar": False},
                            figure=go.Figure(),
                        )
                    ],
                ),
            ],
        ),


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
                html.Div(id="validation-output", className="alert alertError"),
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
    Output("section-fi", "style"),
    Input("model-choice", "value"),
)

def toggle_sections(model_choice):
    show_telaviv = model_choice in ("tel_aviv_v1", "tel_aviv_v2", "tel_aviv_v3_2_clean")
    show_opt = model_choice in ("tel_aviv_v2", "tel_aviv_v3_2_clean")
    show_taiwan = model_choice == "taiwan"

    show_fi = model_choice in ("tel_aviv_v2", "tel_aviv_v3_2_clean")
    def s(show: bool):
        return {"display": "block"} if show else {"display": "none"}

    return s(show_telaviv), s(show_opt), s(show_taiwan), s(show_fi)


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
# Feature importance
# =========================================================
@app.callback(
    Output("feat-importance-graph", "figure"),
    Input("model-choice", "value"),
)
def update_feat_importance(model_choice):
    if model_choice == "tel_aviv_v3_2_clean" and model_tel_aviv_v3_2 is not None:
        cols = tel_aviv_v3_2_feature_cols or tel_aviv_v2_feature_cols or TEL_AVIV_V2_FALLBACK_COLS
        return build_feature_importance_figure(model_tel_aviv_v3_2, cols, top_k=12)

    if model_choice == "tel_aviv_v2" and model_tel_aviv_v2 is not None:
        cols = tel_aviv_v2_feature_cols or TEL_AVIV_V2_FALLBACK_COLS
        return build_feature_importance_figure(model_tel_aviv_v2, cols, top_k=12)

    fig = go.Figure()
    fig.update_layout(
        height=10,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


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

    features = {
        "netArea": netarea,
        "rooms": rooms,
        "floor": floor,
        "constructionYear": year,
        "grossArea": grossarea,
        "floors": floors_total,
        "apartmentsInBuilding": apts,
        "parking": parking,
        "storage": storage,
        "roof": roof,
        "yard": yard,
        "distance": distance,
        "convenience": convenience,
        "lat": lat,
        "long": long_,
    }

    try:
        market_id = "tw-tpe" if model_choice == "taiwan" else "il-tlv"
        result = predict(model_choice, features, market_id)
    except ValueError as e:
        return "", str(e)
    except Exception as e:
        return "", f"Error during prediction: {e}"

    subs = [f"{label}: {fmt_money(lo, result.currency)} – {fmt_money(hi, result.currency)}" for (label, lo, hi) in result.ranges]
    if result.p10 != result.p90:
        subs.append(
            f"Model spread (P10–P90): {fmt_money(result.p10, result.currency)} – {fmt_money(result.p90, result.currency)}"
        )
    subs.append(
        "Note: ranges are approximate; MAE/RMSE come from held-out test data (if available), and model spread comes from RF tree variability (not calibrated)."
    )

    title = "Predicted price" if model_choice == "taiwan" else "Estimated price"
    return ui_result(title, fmt_money(result.price, result.currency), subtitle=subs), ""


if __name__ == "__main__":
    # Чтобы не плодились 2 процесса и не ловить “порт занят”
    port = int(os.environ.get("PORT", "8050"))
    debug = os.environ.get("DEBUG", "0") == "1"
    app.run(debug=debug, use_reloader=False, host="0.0.0.0", port=port)

