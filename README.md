#  Real Estate Price Prediction — Tel Aviv v1/v2 + Taiwan baseline

End-to-end ML project with a clean Dash interface for **real estate price prediction**.  
The project demonstrates a complete workflow from EDA and feature engineering to model versioning and an interactive UI.

> **Focus:** Tel Aviv market (improved model v2) + a Taiwan tutorial baseline for comparison.

---

##  Problem

Real estate pricing depends on many interacting factors: area, floor, building age, building scale, amenities, and market/time effects.  
The goal of this project is to build a practical ML solution that:

- predicts apartment price based on property characteristics,
- improves accuracy via feature engineering,
- provides a clean UI for inference,
- keeps model versions comparable (baseline vs improved).

---

##  Approach

### 1) Data & EDA
- Performed exploratory analysis for both:
  - Tel Aviv dataset
  - Taiwan tutorial dataset
- Checked missing values, distributions, and outliers.

### 2) Feature Engineering (Tel Aviv)

**Core features (examples):**
- `netArea`, `grossArea`, `rooms`, `floor`, `floors`
- `apartmentsInBuilding`, `parking`, `storage`, `roof`, `yard`
- `constructionYear`

**Engineered features:**
- `tx_year`, `tx_month`, `tx_quarter` *(proxy-driven from current date in the app)*
- `building_age_at_tx`
- `floor_ratio`

### 3) Target Transformation
Tel Aviv v2 was trained on:
- `log1p(price)`

The app converts predictions back using:
- `expm1(pred_log)`

### 4) Modeling Strategy
- **Tel Aviv v1** — baseline with minimal inputs
- **Tel Aviv v2** — improved model with expanded feature space + log target
- **Taiwan model** — tutorial baseline

---

##  Results

Below are test-split results from the Tel Aviv experiments (trained on `log(price)`):

| Model | MAE (₪) | RMSE (₪) | R² |
|------|---------:|---------:|----:|
| Linear Regression | 858,171 | 1,778,791 | 0.387 |
| GradientBoostingRegressor | 718,143 | 1,433,070 | 0.602 |
| RandomForest | 673,636 | 1,434,999 | 0.601 |
| ExtraTrees | 659,631 | 1,445,707 | 0.595 |
| HistGBR | 729,013 | 1,485,657 | 0.572 |

**Conclusion:**  
Tree-based methods significantly outperform the linear baseline.  
The **Tel Aviv v2 pipeline** is the strongest version of the project thanks to richer features and log-target training.

---

##  Demo (Dash UI)

Run the app:

```bash
python src/app.py
```
Open:
```
http://127.0.0.1:8050
```


##  Project Structure

```bash
END-TO-END ML PROJECT/
├── data/
│   ├── Real_estate_Tel_Aviv_20_years.csv
│   └── Real_Estate.csv
├── models/
│   ├── real_estate_model.pkl
│   ├── tel_aviv_real_estate_model.pkl
│   ├── tel_aviv_real_estate_model_v2.pkl
│   └── tel_aviv_feature_cols_v2.json
├── notebooks/
│   ├── real_estate_eda.ipynb
│   ├── israel_real_estate_eda.ipynb
│   ├── tel_aviv_model_improvement.ipynb
│   └── israel_tel_aviv_model_improvement.ipynb
├── pics/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.png
│   └── 5.png
├── src/
│   ├── app.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

##  Screenshots

**UI overview + model selector**
![Model selection and Tel Aviv required inputs](pics/1.png)
*Figure 1: Model selection page with Tel Aviv v2/v1 and Taiwan options.*

---

**Tel Aviv v1 — baseline flow**
![Tel Aviv v1 baseline inputs](pics/2.png)
*Figure 2: Tel Aviv v1 baseline model with minimal required inputs.*

---

**Taiwan — tutorial model**
![Taiwan model inputs](pics/3.png)
*Figure 3: Taiwan tutorial model input form (baseline dataset).*

---

**Tel Aviv v2 — prediction result**
![Tel Aviv v2 prediction result](pics/4.png)
*Figure 4: Tel Aviv v2 predicted price in ₪ after reversing log transformation.*

---

**Tel Aviv v2 — optional fields**
![Tel Aviv v2 optional fields](pics/5.png)
*Figure 5: Optional v2 inputs that can improve prediction accuracy.*


## ⚙️ Installation
```
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```


🛠 Tech Stack
```
Python

Pandas / NumPy

Scikit-learn

Joblib

Dash
```

🧭 Roadmap
```
Short-term (portfolio polish):

 Export metrics snapshot to JSON inside models/

 Add feature importance visualization for Tel Aviv v2

 Add a lightweight “Compare v1 vs v2” UX mode

Mid-term (real-world upgrade):

 Try CatBoost / LightGBM / XGBoost

 Cross-validation + error analysis

 Add richer location/geospatial features if available

MLOps-lite:

 Model metadata: training date, dataset hash, metrics

 Simple /predict API wrapper
```

👤 Author

Nikita Marshchonok

ML / Data Science portfolio project.
