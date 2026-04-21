# 🚦 Scalable Traffic Congestion Analysis using CARLA Simulator

> A big data pipeline for synthetic urban traffic analysis — simulation, distributed processing, machine learning, and interactive visualization.

---

## Overview

This project builds an end-to-end traffic congestion prediction system using the **CARLA autonomous driving simulator** as a data source, **Apache PySpark** for distributed preprocessing at scale, and **scikit-learn / XGBoost** for multi-class congestion classification. A **Streamlit dashboard** with SHAP explainability and a continuous severity score brings the results to life interactively.

The system achieves **81.0% accuracy and 0.943 ROC-AUC** (Random Forest) on a held-out test set of ~151K records, trained on over **1 million synthetic vehicle observations** across 4 CARLA towns and 4 weather conditions.

---

## Demo

| Dashboard tab | Preview |
|---|---|
| Data Explorer | Speed distributions, label breakdown, per-session stats |
| Congestion Map | Spatial heatmaps — labels / speed / density / severity score |
| Model Results | Comparison chart, feature importance (RF + XGBoost) |
| Live Predictor | Real-time prediction + SHAP waterfall + severity gauge |

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
carla-traffic-analysis/
│
├── data/
│   ├── raw/                    # Parquet + CSV from CARLA sessions
│   ├── processed/              # Train / val / test splits + feature_cols.txt
│   ├── models_clean/           # Trained model .pkl files + comparison.csv
│   │   └── plots/              # Feature importance, confusion matrices, ROC curves
│   └── visualizations/         # 6 matplotlib figure files (48 panels)
│
├── big_data/
│   ├── carla_runner.py         # CARLA simulation + data collection
│   ├── validate_data.py        # Data quality checks + 9-panel QA report
│   ├── preprocessing.py        # PySpark cleaning, feature engineering, splits
│   ├── train.py                # Logistic Regression, Random Forest, XGBoost
│   ├── visualize.py            # 48-panel matplotlib deep-dive figures
│   └── dashboard/
│       └── app.py              # Streamlit dashboard
│
└── README.md
```

---

## System Architecture

```
CARLA Simulator
  └─ carla_runner.py          → data/raw/*.parquet  (1M+ rows)
       │
       ├─ validate_data.py    → QA report + 9-panel PNG
       │
       └─ preprocessing.py   → PySpark clean + feature eng + stratified split
            │
            └─ train.py       → RF / XGBoost / LogReg → .pkl models
                 │
                 ├─ visualize.py       → 48-panel matplotlib figures
                 └─ dashboard/app.py  → Streamlit interactive dashboard
```

---

## Dataset

| Property | Value |
|---|---|
| Total records | 1,007,762 |
| Sessions | 8 |
| Towns | Town01, Town02, Town03, Town04 |
| Weather conditions | ClearNoon, HardRainNoon, MidRainyNoon, WetCloudyNoon |
| Vehicle count range | 60 – 150 per session |
| Simulation FPS | 20 |
| Storage format | Parquet (Snappy compressed) + CSV |

### Schema

| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique run identifier (timestamp + town + weather) |
| `tick` | int | Simulation tick number |
| `vehicle_id` | int | CARLA actor ID |
| `speed` | float | Vehicle speed (km/h) |
| `acceleration` | float | Acceleration magnitude (m/s²) |
| `x`, `y`, `z` | float | World coordinates (metres) |
| `yaw` | float | Vehicle heading (degrees) |
| `traffic_density` | int | Vehicles within 50m radius |
| `weather` | int | Encoded weather condition (0–20) |
| `weather_name` | string | Human-readable weather preset |
| `town` | string | CARLA map name |
| `hour` | int | Simulated hour of day (0–23) |
| `congestion_label` | string | Target: Low / Medium / High |

### Congestion label definition

Labels are assigned per vehicle per tick using a density + speed interaction rule:

```
High   → density ≥ 10 vehicles/50m  AND  speed < 20 km/h
Low    → density < 4  OR  speed ≥ 50 km/h
Medium → everything else
```

Label distribution across the full dataset: **Low 34.6% · Medium 36.2% · High 29.2%**

---

## Setup

### Prerequisites

- Ubuntu 20.04 / 22.04
- CARLA 0.9.16 ([download](https://github.com/carlaSimulator/carla/releases))
- Python 3.10+
- Java 11+ (required for PySpark)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/carla-traffic-analysis.git
cd carla-traffic-analysis

# Create virtual environment
python -m venv carla_env
source carla_env/bin/activate

# Install dependencies
pip install carla pyarrow pandas pyspark \
            scikit-learn xgboost shap \
            streamlit plotly matplotlib seaborn scipy
```

---

## Usage

### Step 1 — Run simulation and collect data

Edit the config block at the top of `carla_runner.py`, then with CARLA server running:

```bash
# Start CARLA server first (in a separate terminal)
./CarlaUE4.sh -RenderOffScreen

# Collect one session
python big_data/carla_runner.py
```

Key config options:

```python
TOWN           = "Town03"        # Town01–Town05, Town10HD
VEHICLE_COUNT  = 80              # 50–200
WEATHER_PRESET = "ClearNoon"     # see full list in script
HOUR_OF_DAY    = 12              # logged as feature
DURATION_SECS  = 300             # simulation seconds
```

Repeat for different towns / weathers to build dataset diversity. Each run writes a uniquely named `.parquet` + `.csv` to `data/raw/`.

### Step 2 — Validate data quality

```bash
python big_data/validate_data.py
```

Runs 8 checks (schema, volume, nulls, duplicates, value ranges, label balance, spatial coverage, speed-density correlation) and saves a 9-panel PNG to `data/validation/`.

```
FINAL VERDICT
[PASS] Data looks good — ready for PySpark preprocessing.
```

### Step 3 — Preprocess with PySpark

```bash
python big_data/preprocessing.py
```

Performs distributed cleaning, feature engineering, label encoding, and stratified 70/15/15 train/val/test split. Outputs Parquet splits to `data/processed/`.

Engineered features added on top of raw schema:

| Feature | Description |
|---|---|
| `speed_bin` | Ordinal speed bracket (0–3) |
| `density_bin` | Ordinal density bracket (0–3) |
| `is_stationary` | Speed < 1 km/h |
| `is_high_density` | Density ≥ 10 vehicles/50m |
| `is_rush_hour` | Hour in 7–9 or 17–19 |
| `is_rainy` | Weather maps to wet condition |
| `speed_x_density` | Interaction term |
| `is_braking` | High accel + low speed proxy |
| `town_code` | Integer extracted from town name |

> **Leakage note:** `traffic_density`, `density_bin`, `is_high_density`, and `speed_x_density` are excluded from model training because the congestion label is a deterministic function of `traffic_density` + `speed`. Including them inflates accuracy to 1.000 trivially. The leakage-free 11-feature model is the production version.

### Step 4 — Train models

```bash
python big_data/train.py
```

Trains Logistic Regression, Random Forest (200 trees), and XGBoost (early stopping on val). Saves `.pkl` models and plots to `data/models_clean/`.

### Step 5 — Visualize

```bash
python big_data/visualize.py
```

Generates 6 deep-dive figures (48 panels total) to `data/visualizations/`.

### Step 6 — Launch dashboard

```bash
streamlit run big_data/dashboard/app.py
```

Opens at `http://localhost:8501`.

---

## Results

### Model comparison — test set (leakage-free, 11 features)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.614 | 0.630 | 0.614 | 0.610 | 0.798 |
| XGBoost | 0.776 | 0.784 | 0.776 | 0.777 | 0.920 |
| **Random Forest** ✓ | **0.810** | **0.814** | **0.810** | **0.810** | **0.943** |

### Confusion matrix highlights (Random Forest)

| Actual → Predicted | Low | Medium | High |
|---|---|---|---|
| **Low** | **78.4%** | 20.2% | 1.4% |
| **Medium** | 9.3% | **76.4%** | 14.3% |
| **High** | 0.4% | 9.8% | **89.8%** |

High congestion is the easiest class (89.8%) — stationary / near-stationary vehicles are a strong unambiguous signal. Medium is hardest (76.4%) as it sits between two well-separated states.

### Key feature importances (Random Forest)

`speed` (0.23) > `speed_bin` (0.22) > `yaw` (0.21) > `town_code` (0.16) > `is_stationary` (0.06)

`yaw` ranking 3rd is notable — vehicles in congested intersections have erratic headings compared to highway free-flow, which the model learned without being told.

---

## Dashboard Features

### Congestion severity score

A continuous 0–100 score derived from class probabilities — no retraining required:

```
severity = 0×P(Low) + 50×P(Medium) + 100×P(High)
```

This replaces the discrete 3-class label with a smooth gradient, enabling the severity heatmap view and the gauge in the live predictor.

### SHAP explainability

Every prediction in the Live Predictor tab generates a **SHAP waterfall chart** showing which features pushed the model toward or away from the predicted class, plus a 5-line plain-English summary. Uses `shap.TreeExplainer` on Random Forest and XGBoost.

---

## Big Data Justification (5Vs)

| V | Description |
|---|---|
| **Volume** | 1M+ synthetic vehicle records across 8 sessions |
| **Velocity** | 20 FPS simulation → high-frequency vehicle state updates |
| **Variety** | 4 towns, 4 weather conditions, 16 feature dimensions |
| **Veracity** | Controlled simulation removes sensor noise and labelling ambiguity |
| **Value** | Congestion predictions + spatial heatmaps actionable for smart city planning |

---

## Limitations & Future Work

- **Covariate shift** — model trained on CARLA autopilot behaviour; real traffic has pedestrians, lane changes, and irregular driving patterns not present in simulation
- **Static hour** — `HOUR_OF_DAY` was hardcoded to 12 across all sessions; temporal variation is absent from the dataset
- **Fixed density radius** — 50m radius for density computation is constant; road-segment-aware density would be more physically meaningful
- **Potential extensions** — DAgger / online learning loop to reduce covariate shift; LSTM forecasting on tick sequences; HERE Traffic API integration for real-world validation; Spark MLlib training for a fully distributed pipeline

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | CARLA 0.9.16 (Python API) |
| Data storage | Apache Parquet (PyArrow + Snappy) |
| Distributed processing | Apache PySpark 3.x (local mode) |
| Machine learning | scikit-learn 1.x, XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Language | Python 3.10 |

---

## Resume Summary

> Built a scalable traffic congestion prediction system using CARLA simulator and PySpark, generating 1M+ synthetic driving records across 4 towns and 4 weather conditions. Trained leakage-free Random Forest and XGBoost classifiers achieving **81% accuracy and 0.943 ROC-AUC**. Deployed an interactive Streamlit dashboard with SHAP explainability, a continuous severity score, and spatial congestion heatmaps.

---

## Acknowledgements

- [CARLA Simulator](https://carla.org/) — CARLA Team, Computer Vision Center
- [Apache Spark](https://spark.apache.org/) — Apache Software Foundation
- [SHAP](https://github.com/shap/shap) — Scott Lundberg et al.
