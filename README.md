# Real Estate Price Prediction — Tel Aviv (v1/v2/v3.2_clean) + Taiwan Baseline

End-to-end ML project with a clean Dash interface for **real estate price prediction**.  
The project demonstrates a complete workflow from EDA and feature engineering to model versioning, leakage-aware evaluation, and an interactive UI.

> **Focus:** Tel Aviv market (best model: **v3.2_clean**) + a Taiwan tutorial baseline for comparison.

---

## Problem

Real estate pricing depends on many interacting factors: area, floor, building age, building scale, amenities, and market/time effects.  
The goal of this project is to build a practical ML solution that:

- predicts apartment price based on property characteristics,
- improves accuracy via feature engineering + model tuning,
- avoids leakage via a **time-aware split**,
- provides a clean UI for inference,
- keeps model versions comparable (baseline vs improved).

---

## Approach

### 1) Data & EDA
- Performed exploratory analysis for:
  - Tel Aviv dataset
  - Taiwan tutorial dataset
- Checked missing values, distributions, and outliers.

### 2) Feature Engineering (Tel Aviv)

**Core features:**
- `netArea`, `grossArea`, `rooms`, `floor`, `floors`
- `apartmentsInBuilding`, `parking`, `storage`, `roof`, `yard`
- `constructionYear`

**Engineered features:**
- `tx_year`, `tx_month`, `tx_quarter`
- `building_age_at_tx`
- `floor_ratio`

### 3) Target Transformation
Tel Aviv models are trained on:
- `log1p(price)`

The app converts predictions back using:
- `expm1(pred_log)`

### 4) Modeling Strategy
- **Tel Aviv v1** — baseline with minimal inputs
- **Tel Aviv v2** — expanded feature space + log target
- **Tel Aviv v3 / v3.2_clean** — tuned RandomForest + missing indicators + production-style evaluation
- **Taiwan model** — tutorial baseline

---

## Results (Tel Aviv)

### Quick comparison (classic split)
These are quick baseline experiments on a regular holdout split (trained on `log1p(price)`):

| Model | MAE (₪) | RMSE (₪) | R² |
|------|---------:|---------:|----:|
| Linear Regression | 858,171 | 1,778,791 | 0.387 |
| GradientBoostingRegressor | 718,143 | 1,433,070 | 0.602 |
| RandomForest | 673,636 | 1,434,999 | 0.601 |
| ExtraTrees | 659,631 | 1,445,707 | 0.595 |
| HistGBR | 729,013 | 1,485,657 | 0.572 |

### Production-style evaluation (time-aware split)
Evaluation is performed on a **time-aware split** to avoid leakage:
- **Train:** before `2018-03-25`
- **Test:** on/after `2018-03-25`

| Version | Model | Split | MAE (₪) | RMSE (₪) | R² |
|---|---|---:|---:|---:|---:|
| v2 (FAIR baseline) | RF(500) + median imputer | Time-aware | 1,076,070 | 1,618,660 | 0.5113 |
| v3 | Tuned RF + missing indicators | Time-aware | 1,025,612 | 1,576,916 | 0.5362 |
| **v3.2_clean (BEST)** | Tuned RF + missing indicators + **data cleaning** | Time-aware | **928,568** | **1,503,636** | **0.5754** |

#### Data cleaning impact
- v3 MAE (before cleaning): **1,025,612 ₪**
- v3.2_clean MAE (after cleaning): **928,568 ₪**
- Improvement: **97,044 ₪**

#### Error analysis (high-level)
- Errors grow with **price and area** (luxury / rare properties are harder).
- The model tends to underestimate high-end properties (typical regression-to-the-mean behavior).
- Fixed/filtered issues: `constructionYear > transaction year`, invalid/zero areas, extreme prices.

---

## Model Insights — Permutation Importance (v3.2_clean)

Top drivers: **netArea, grossArea, constructionYear, building_age_at_tx, rooms, floors/floor, parking**.

![Permutation Importance](pics/tel_aviv_v3_2_perm_importance.png)

---

## Demo (Dash UI)

Run the app:

```bash
python src/app.py
