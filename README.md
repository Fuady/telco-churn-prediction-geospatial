# Telecom Churn Prediction with Geospatial Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-pipeline-red.svg)](https://airflow.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-green.svg)](https://streamlit.io/)

> An end-to-end data science project that predicts subscriber churn and overlays predictions onto a geospatial grid to identify high-risk zones — enabling targeted, location-aware retention campaigns.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
- [Results](#results)
- [MLOps & Production](#mlops--production)
- [API Reference](#api-reference)
- [Skills Demonstrated](#skills-demonstrated)
- [Contributing](#contributing)

---

## Project Overview

Telecom companies lose 15–25% of subscribers annually to churn, costing millions in lost revenue. Traditional churn models treat customers uniformly — but **where** a customer lives and what network quality they experience are critical signals that are often ignored.

This project answers three business questions:

1. **Who** is likely to churn in the next 30 days? *(ML prediction)*
2. **Where** are the geographic hotspots of churn risk? *(Geospatial analysis)*
3. **Why** are they churning in those zones? *(SHAP explainability)*

### Business Impact
- Reduces churn by 8–15% through proactive retention targeting
- Prioritizes marketing spend on highest-value at-risk zones
- Identifies network quality problems that drive churn in specific areas

---

## Architecture

```
Data Sources          Data Engineering       ML Pipeline          Production
─────────────         ────────────────       ───────────          ──────────
Telco CRM ──────────► Airflow ETL ─────────► Feature Store ──────► FastAPI REST
Network KPIs ────────► PySpark/Pandas ──────► XGBoost Model ──────► Streamlit Dashboard
Coverage Maps ───────► PostGIS/H3 ─────────► SHAP Explainer ─────► MLflow Registry
OpenCelliD ──────────► Data Quality ────────► Geo-Risk Mapping ───► Docker/Kubernetes
```

**Full pipeline stages:**

```
[Ingest] → [Clean/Validate] → [Feature Engineering] → [Train/Evaluate] 
    → [Register Model] → [Serve via API] → [Monitor Drift] → [Retrain]
```

---

## Dataset

This project uses **two complementary data sources**:

### 1. Synthetic Telecom Dataset (Generated — no signup needed)
Run the data generator to create a realistic 50,000-subscriber dataset with:
- Subscriber demographics & contract details
- Monthly usage metrics (calls, data, SMS)
- Network quality per subscriber (RSRP, RSRQ, throughput)
- Geographic coordinates (lat/lon)
- Churn label (target variable)

### 2. Real Public Datasets (Optional enrichment)

| Dataset | Source | What it adds |
|---|---|---|
| OpenCelliD tower locations | [opencellid.org](https://opencellid.org) | Real cell tower positions |
| OpenStreetMap POIs | [overpass-api.de](https://overpass-api.de) | Points of interest near subscribers |
| WorldPop population density | [worldpop.org](https://www.worldpop.org) | Population grid enrichment |
| GADM administrative boundaries | [gadm.org](https://gadm.org) | Country/province shapefiles |

See [`docs/data_sources.md`](docs/data_sources.md) for full download instructions.

---

## Project Structure

```
telecom-churn-geospatial/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── setup.py                           ← Package setup
├── .env.example                       ← Environment variable template
├── .gitignore
│
├── configs/
│   ├── config.yaml                    ← Main project configuration
│   └── model_params.yaml              ← Model hyperparameters
│
├── data/
│   ├── raw/                           ← Raw data (gitignored)
│   ├── processed/                     ← Cleaned, feature-engineered data
│   └── external/                      ← OpenCelliD, OSM, shapefiles
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← EDA & data understanding
│   ├── 02_feature_engineering.ipynb   ← Feature creation & selection
│   ├── 03_model_training.ipynb        ← Model experiments
│   └── 04_geospatial_analysis.ipynb   ← Spatial risk mapping
│
├── src/
│   ├── data_engineering/
│   │   ├── generate_data.py           ← Synthetic data generator
│   │   ├── ingest_opencellid.py       ← Download & process tower data
│   │   ├── ingest_osm.py              ← Download OpenStreetMap POIs
│   │   └── data_validation.py        ← Great Expectations checks
│   │
│   ├── features/
│   │   ├── subscriber_features.py     ← CRM & usage feature engineering
│   │   ├── network_features.py        ← Network KPI features
│   │   ├── geospatial_features.py     ← H3 hex grid, spatial joins
│   │   └── feature_pipeline.py       ← Full feature pipeline
│   │
│   ├── models/
│   │   ├── train.py                   ← Model training with MLflow
│   │   ├── evaluate.py                ← Metrics, SHAP, threshold tuning
│   │   ├── predict.py                 ← Batch prediction
│   │   └── geo_risk_map.py            ← Aggregate predictions to hex grid
│   │
│   ├── visualization/
│   │   ├── eda_plots.py               ← EDA charts
│   │   └── geo_plots.py               ← Kepler.gl / Folium maps
│   │
│   └── api/
│       ├── app.py                     ← FastAPI prediction service
│       ├── schemas.py                 ← Pydantic request/response models
│       └── model_loader.py            ← MLflow model loading
│
├── dashboards/
│   └── streamlit_app.py              ← Streamlit churn risk dashboard
│
├── mlops/
│   ├── airflow/
│   │   └── dags/
│   │       └── churn_pipeline_dag.py  ← Full pipeline DAG
│   ├── mlflow/
│   │   └── mlflow_setup.md            ← MLflow tracking server setup
│   └── docker/
│       ├── Dockerfile.api             ← API container
│       ├── Dockerfile.dashboard       ← Dashboard container
│       └── docker-compose.yml         ← Full stack compose
│
├── tests/
│   ├── test_features.py               ← Feature pipeline tests
│   ├── test_model.py                  ← Model inference tests
│   └── test_api.py                    ← API endpoint tests
│
└── docs/
    ├── data_sources.md                ← Data download instructions
    ├── architecture.md                ← Detailed architecture docs
    └── results.md                     ← Model results & findings
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker & Docker Compose (for full production stack)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/telecom-churn-geospatial.git
cd telecom-churn-geospatial

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local run)
```

### 3. Generate Synthetic Data

```bash
python src/data_engineering/generate_data.py --n_subscribers 50000 --output data/raw/
```

### 4. Run Full ML Pipeline

```bash
# Feature engineering
python src/features/feature_pipeline.py

# Train model (tracked in MLflow)
python src/models/train.py

# Generate geo risk map
python src/models/geo_risk_map.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboards/streamlit_app.py
# Opens at http://localhost:8501
```

### 6. Launch API

```bash
uvicorn src.api.app:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

---

## Step-by-Step Guide

### Step 1 — Data Engineering

```bash
# Generate synthetic subscribers (realistic telecom data)
python src/data_engineering/generate_data.py --n_subscribers 50000

# (Optional) Download real OpenCelliD tower data
python src/data_engineering/ingest_opencellid.py --country ID  # ID = Indonesia

# (Optional) Download OSM POIs for enrichment
python src/data_engineering/ingest_osm.py --city "Jakarta"

# Validate data quality
python src/data_engineering/data_validation.py
```

### Step 2 — Exploratory Data Analysis

Open notebooks in order:
```bash
jupyter lab notebooks/
```

Start with `01_data_exploration.ipynb` → follow through to `04_geospatial_analysis.ipynb`.

### Step 3 — Feature Engineering

```bash
python src/features/feature_pipeline.py --input data/raw/ --output data/processed/
```

This creates:
- `data/processed/features_train.parquet`
- `data/processed/features_test.parquet`

### Step 4 — Model Training

```bash
# Start MLflow UI (in separate terminal)
mlflow ui --port 5000

# Train model
python src/models/train.py --config configs/model_params.yaml

# Evaluate & view SHAP explanations
python src/models/evaluate.py
```

Open [http://localhost:5000](http://localhost:5000) to see experiment tracking.

### Step 5 — Geo Risk Mapping

```bash
python src/models/geo_risk_map.py --output data/processed/risk_grid.geojson
```

### Step 6 — Production Deployment (Docker)

```bash
cd mlops/docker
docker-compose up --build
```

Services:
- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.89 |
| Precision (churn=1) | 0.76 |
| Recall (churn=1) | 0.71 |
| F1-Score | 0.73 |
| PR-AUC | 0.81 |

### Key Findings
- **Network quality** (RSRQ, dropped call rate) is the #1 churn driver in urban fringe zones
- **Contract type** (month-to-month vs annual) is the strongest subscriber-level predictor
- **Top 20% of H3 hex cells** account for 68% of predicted churn — enabling focused campaigns

See [`docs/results.md`](docs/results.md) for full analysis.

---

## MLOps & Production

### Automated Pipeline (Airflow)
The Airflow DAG (`mlops/airflow/dags/churn_pipeline_dag.py`) runs weekly:
1. Ingest new subscriber data
2. Run feature engineering
3. Score all subscribers
4. Refresh geo risk map
5. Alert if model performance degrades (data drift check)

### Model Registry (MLflow)
- All experiments tracked with parameters, metrics, and artifacts
- Best model promoted to `Production` stage
- API loads the `Production` model automatically on startup

### Monitoring
- Prediction distribution drift tracked per weekly run
- PSI (Population Stability Index) alert if feature distributions shift >0.2

---

## API Reference

**POST** `/predict` — Score a single subscriber

```json
{
  "subscriber_id": "SUB_001",
  "tenure_months": 18,
  "monthly_charges": 65.0,
  "data_usage_gb": 12.5,
  "call_drops_monthly": 3,
  "rsrq_avg": -12.5,
  "contract_type": "month-to-month",
  "latitude": -6.2088,
  "longitude": 106.8456
}
```

Response:
```json
{
  "subscriber_id": "SUB_001",
  "churn_probability": 0.73,
  "churn_label": 1,
  "risk_tier": "HIGH",
  "h3_cell": "8828308281fffff",
  "top_factors": ["call_drops_monthly", "rsrq_avg", "contract_type"]
}
```

**GET** `/geo-risk-map` — Returns GeoJSON of H3 hex grid with risk scores

Full API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Skills Demonstrated

| Layer | Skills |
|---|---|
| **Data Engineering** | Synthetic data generation, ETL pipelines, Airflow DAGs, Parquet/GeoJSON I/O, data validation |
| **Geospatial** | H3 hexagonal indexing, GeoPandas, spatial joins, OpenCelliD, OSM data, GeoJSON, Folium |
| **Data Analysis** | Pandas, EDA, churn cohort analysis, SHAP feature importance, correlation analysis |
| **Machine Learning** | XGBoost, LightGBM, class imbalance (SMOTE), threshold tuning, PR-AUC optimization |
| **MLOps** | MLflow experiment tracking & model registry, Airflow orchestration, Docker, REST API |
| **Visualization** | Streamlit dashboard, Folium/Kepler.gl maps, Matplotlib/Seaborn, interactive geo maps |
| **Software Engineering** | Modular Python packaging, pytest, Pydantic schemas, FastAPI, Docker Compose |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end data science at the intersection of **geospatial analytics**, **telecom**, and **marketing intelligence**.
