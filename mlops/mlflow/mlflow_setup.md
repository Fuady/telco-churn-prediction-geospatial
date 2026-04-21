# MLflow Setup Guide

This guide explains how to set up MLflow tracking for this project.

---

## Local Development Setup

### 1. Start the MLflow Tracking Server

```bash
# Install MLflow
pip install mlflow==2.12.2

# Start the UI (in a separate terminal — keep it running)
mlflow ui --host 0.0.0.0 --port 5000

# Open the UI in your browser:
# http://localhost:5000
```

### 2. Run Training

MLflow tracking is automatic — just run training normally:

```bash
python src/models/train.py
```

You'll see logs like:
```
MLflow run ID: a1b2c3d4e5f6...
Experiment: telecom_churn_prediction
```

Open http://localhost:5000 to see:
- All experiment runs with parameters and metrics
- Artifact charts (feature importance, SHAP plots)
- Model comparison table

---

## Experiment Tracking Details

### What Gets Logged Per Run

| Category | Items Logged |
|----------|-------------|
| Parameters | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`, `model_type`, `n_features`, `train_size` |
| Metrics | `roc_auc`, `pr_auc`, `f1`, `precision`, `recall`, `threshold` |
| Artifacts | `feature_importance.png`, `shap_summary.png`, `churn_model_{type}.pkl`, `feature_columns.txt` |
| Tags | Run name: `{model_type}_churn` |
| Model | Registered as `churn_xgboost` in the Model Registry |

### Model Registry Stages

```
None → Staging → Production → Archived
```

After training, promote the best model to Production:

```python
import mlflow
client = mlflow.tracking.MlflowClient()

# List versions
for v in client.search_model_versions("name='churn_xgboost'"):
    print(f"Version {v.version}: {v.current_stage} | run_id: {v.run_id[:8]}")

# Promote version 3 to Production
client.transition_model_version_stage(
    name="churn_xgboost",
    version="3",
    stage="Production",
)
```

Or via the MLflow UI: Models → churn_xgboost → (version) → Stage: Production

---

## Remote Tracking Server (Production)

For a team setup, run MLflow on a remote server with a PostgreSQL backend:

```bash
# On the remote server
pip install mlflow psycopg2-binary boto3

# With PostgreSQL backend + S3 artifact store
mlflow server \
  --backend-store-uri postgresql://user:pass@db-host/mlflow_db \
  --default-artifact-root s3://your-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Update `.env`:
```
MLFLOW_TRACKING_URI=http://your-server:5000
```

---

## Comparing Runs Programmatically

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# Get all runs for an experiment
runs = mlflow.search_runs(
    experiment_names=["telecom_churn_prediction"],
    order_by=["metrics.pr_auc DESC"],
)

# Top 5 runs by PR-AUC
cols = ["run_id", "metrics.pr_auc", "metrics.roc_auc", "metrics.f1",
        "params.model_type", "params.n_estimators"]
print(runs[cols].head(5).to_string())
```

---

## Loading the Production Model in the API

The API (`src/api/model_loader.py`) automatically loads the `Production` stage model:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/churn_xgboost/Production")
predictions = model.predict(X_new)
```

---

## Experiment Structure

```
telecom_churn_prediction/          ← Experiment
├── run: xgboost_churn (v1.0)
│   ├── params/
│   ├── metrics/
│   └── artifacts/
│       ├── feature_importance.png
│       ├── shap_summary.png
│       └── model/
├── run: lightgbm_churn (v1.1)
└── run: xgboost_optuna (v1.3)    ← Best model → Production
```

---

## Useful MLflow CLI Commands

```bash
# List experiments
mlflow experiments list

# Get best run
mlflow runs list --experiment-name telecom_churn_prediction

# Download artifacts from a run
mlflow artifacts download -r RUN_ID -d ./downloaded_artifacts

# Serve the Production model as a REST API (alternative to FastAPI)
mlflow models serve -m "models:/churn_xgboost/Production" --port 9000
# POST http://localhost:9000/invocations
```
