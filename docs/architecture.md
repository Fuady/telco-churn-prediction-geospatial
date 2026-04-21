# Architecture Documentation

## System Overview

This project implements a full end-to-end ML platform for telecom churn prediction with geospatial risk mapping. The architecture follows the **Lambda pattern**: a batch training pipeline that refreshes weekly, and a real-time serving layer that provides instant predictions.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                    │
│                                                                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │  Telecom CRM    │  │  OpenCelliD      │  │  OpenStreetMap (OSM)    │ │
│  │  (subscriber    │  │  (cell towers,   │  │  (POIs, road network,   │ │
│  │  demographics,  │  │  signal data,    │  │  commercial zones,      │ │
│  │  usage, billing)│  │  free API)       │  │  free Overpass API)     │ │
│  └────────┬────────┘  └────────┬─────────┘  └────────────┬────────────┘ │
└───────────┼─────────────────────┼───────────────────────────┼────────────┘
            │                     │                           │
            ▼                     ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       DATA ENGINEERING LAYER                             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Apache Airflow DAG (weekly)                    │   │
│  │                                                                   │   │
│  │  generate_data.py ──► ingest_opencellid.py ──► ingest_osm.py    │   │
│  │       │                       │                      │           │   │
│  │       └───────────────────────┴──────────────────────┘           │   │
│  │                               │                                   │   │
│  │                    data_validation.py                             │   │
│  │                    (Great Expectations checks)                    │   │
│  └────────────────────────────┬──────────────────────────────────────┘  │
│                                │                                          │
│                       data/raw/ (Parquet)                                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                           │
│                                                                          │
│  subscriber_features.py     network_features.py    geospatial_features  │
│  ─────────────────────      ────────────────────   ────────────────────  │
│  • tenure buckets           • RSRQ/RSRP quality    • H3 indexing (r7,r8) │
│  • charge ratios            • quality score (0-100) • tower proximity    │
│  • contract risk score      • frustration index    • POI distances       │
│  • support intensity        • drop rate buckets    • H3 aggregates       │
│                                                                          │
│                    feature_pipeline.py                                   │
│                    (orchestrates all transformers)                       │
│                                                                          │
│               data/processed/features_{train,test,full}.parquet         │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  train.py                                                        │    │
│  │  ─────────                                                       │    │
│  │  1. SMOTE (class balance)                                        │    │
│  │  2. XGBoost / LightGBM training                                  │    │
│  │  3. Optuna HPO (optional --tune flag)                            │    │
│  │  4. Threshold tuning (maximise F1)                               │    │
│  │  5. MLflow logging (params, metrics, artifacts)                  │    │
│  │  6. Model registration to MLflow Registry                        │    │
│  └─────────────────────┬───────────────────────────────────────────┘    │
│                         │                                                │
│                  ┌──────┴──────┐                                        │
│                  │  evaluate.py│  ← confusion matrix, ROC, PR, SHAP,   │
│                  └──────┬──────┘    lift curve, calibration             │
│                         │                                                │
│                  ┌──────┴──────┐                                        │
│                  │geo_risk_map │  ← H3 aggregation → GeoJSON + Folium  │
│                  └─────────────┘                                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                    ┌───────────┴──────────┐
                    │    MLflow Registry    │
                    │  (model versioning)   │
                    │  None→Staging→       │
                    │  Production→Archived  │
                    └───────────┬──────────┘
                                │
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   FastAPI (REST) │ │ Streamlit        │ │ Batch predict.py │
│   ─────────────  │ │ Dashboard        │ │ ────────────────  │
│  POST /predict   │ │ ──────────────── │ │ Weekly scoring   │
│  POST /predict/  │ │ • KPI overview   │ │ of all subs      │
│    batch         │ │ • Folium geo map │ │ → CSV/Parquet    │
│  GET /geo-risk   │ │ • Subscriber     │ │   export         │
│  GET /health     │ │   lookup + gauge │ │                  │
│                  │ │ • SHAP plots     │ │                  │
│  Port: 8000      │ │ • Alerts table   │ │                  │
└──────────────────┘ │                  │ └──────────────────┘
                     │  Port: 8501      │
                     └──────────────────┘

───────────────────── MONITORING ─────────────────────────────────────────

  Airflow DAG (weekly):
    data_validation → features → batch_predict → geo_refresh
                                                      │
                                                 drift_check (PSI)
                                                 /            \
                                            retrain         skip
                                                      │
                                                 notify (Slack)
```

---

## Component Responsibilities

### Data Engineering (`src/data_engineering/`)

| Module | Responsibility |
|--------|---------------|
| `generate_data.py` | Creates a realistic 50k-subscriber synthetic dataset with proper churn correlations. Used for demo/dev. |
| `ingest_opencellid.py` | Downloads and filters real cell tower data by country MCC code. Handles gzip decompression and chunked CSV reading. |
| `ingest_osm.py` | Pulls POI data from Overpass API (free, no key). Rate-limited to be polite to the public API. |
| `data_validation.py` | Lightweight Great Expectations-style checks. Fails pipeline fast on data quality issues. |

### Feature Engineering (`src/features/`)

| Module | Responsibility |
|--------|---------------|
| `subscriber_features.py` | sklearn-compatible transformer for CRM/usage features. Stateless — `fit()` is a no-op. |
| `network_features.py` | Transforms raw RSRP/RSRQ/throughput into quality scores and categories. |
| `geospatial_features.py` | H3 indexing, vectorised haversine distance (chunk-processed for memory efficiency), spatial joins. |
| `feature_pipeline.py` | Orchestrates all transformers. Outputs train/test Parquet splits + feature list. |

### Modelling (`src/models/`)

| Module | Responsibility |
|--------|---------------|
| `train.py` | Full training loop: SMOTE → model fit → threshold tuning → MLflow logging → model registration. |
| `evaluate.py` | Standalone evaluation producing ROC/PR/confusion matrix/lift/calibration charts. |
| `predict.py` | Batch scoring of new data. Supports both raw input (triggers feature engineering) and pre-processed input. |
| `geo_risk_map.py` | Aggregates subscriber-level predictions to H3 grid. Outputs GeoJSON + interactive Folium map. |

### API (`src/api/`)

| Module | Responsibility |
|--------|---------------|
| `app.py` | FastAPI application with CORS, startup model loading, and 5 endpoints. |
| `schemas.py` | Pydantic v2 input/output models with field-level validation. |
| `model_loader.py` | Singleton model loader — tries MLflow first, falls back to local pickle. Runs the same feature engineering as training. |

---

## Data Flow

```
Subscriber raw data
        │
        ▼
[Validation] ──► fail fast if data quality issues
        │
        ▼
[Feature Engineering] ──► 59 features per subscriber
        │
        ▼
[Model Scoring] ──► churn_probability (0–1) per subscriber
        │
        ├──► [API] ──► Real-time single/batch predictions
        │
        ├──► [H3 Aggregation] ──► Risk map per H3 cell
        │
        └──► [Dashboard] ──► Visual analytics + alerts
```

---

## Technology Choices

| Choice | Rationale |
|--------|-----------|
| **XGBoost** over deep learning | Tree models outperform DNNs on tabular data with <100k rows (benchmark: Grinsztajn et al. 2022). Faster, interpretable via SHAP. |
| **SMOTE** over `class_weight` | Oversampling the minority class before training outperforms weight-based approaches on PR-AUC for imbalanced telecom data. |
| **H3 res=8** for risk map | 0.74 km² — approximately city-block size. Coarse enough for ≥10 subscribers/cell, fine enough for targeted marketing. |
| **PR-AUC** as primary metric | ROC-AUC is optimistic on imbalanced data. PR-AUC reflects precision–recall trade-off at all thresholds — more meaningful for marketing budget allocation. |
| **FastAPI** over Flask | Async-capable, native Pydantic v2 validation, auto-generated OpenAPI docs, 3× higher throughput on concurrent requests. |
| **Airflow** over Cron | DAG-based pipeline enables retry logic, branching (conditional retrain), monitoring, and alerting — not possible with plain cron. |
| **MLflow** over custom logging | Industry-standard experiment tracking. Provides model registry with staging/production lifecycle management. |

---

## Scaling Considerations

| Scenario | Current approach | Scale-up path |
|----------|-----------------|--------------|
| 50k subscribers | Pandas in-memory | ✅ Sufficient |
| 5M subscribers | Pandas struggles | PySpark + Delta Lake |
| 50M subscribers (CDR) | Out of memory | Apache Spark + Kafka streams |
| Real-time scoring (<100ms) | FastAPI + pickle | FastAPI + ONNX-quantised model |
| Multi-country | Single config | Parameterise by country MCC in Airflow |
