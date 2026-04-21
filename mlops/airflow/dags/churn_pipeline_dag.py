"""
mlops/airflow/dags/churn_pipeline_dag.py
──────────────────────────────────────────
Weekly automated pipeline DAG for churn prediction refresh.

Schedule: Every Sunday at 02:00 UTC
Pipeline:
  1. data_validation     — check raw data quality
  2. feature_engineering — rebuild feature store
  3. batch_prediction    — score all subscribers
  4. geo_risk_refresh    — update H3 risk map
  5. drift_detection     — check for data drift
  6. conditional_retrain — retrain if drift detected
  7. notify              — send summary to Slack/email

To set up Airflow locally:
  pip install apache-airflow
  export AIRFLOW_HOME=~/airflow
  airflow db init
  airflow webserver --port 8080 &
  airflow scheduler &
  # Copy this file to ~/airflow/dags/
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ── DAG definition ────────────────────────────────────────────────────────────
dag = DAG(
    dag_id="telecom_churn_pipeline",
    default_args=default_args,
    description="Weekly churn prediction pipeline with geo risk mapping",
    schedule_interval="0 2 * * 0",  # Every Sunday at 02:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["churn", "geospatial", "ml", "production"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def run_data_validation(**context):
    """Validate raw data quality before processing."""
    import subprocess
    result = subprocess.run(
        ["python", "src/data_engineering/data_validation.py",
         "--input", "data/raw/subscribers.parquet"],
        capture_output=True, text=True, cwd="/app"
    )
    if result.returncode != 0:
        raise ValueError(f"Data validation failed:\n{result.stderr}")

    context["task_instance"].xcom_push(key="validation_passed", value=True)


def run_feature_engineering(**context):
    """Run full feature engineering pipeline."""
    import subprocess
    result = subprocess.run(
        ["python", "src/features/feature_pipeline.py",
         "--input", "data/raw",
         "--output", "data/processed"],
        capture_output=True, text=True, cwd="/app"
    )
    if result.returncode != 0:
        raise ValueError(f"Feature engineering failed:\n{result.stderr}")


def run_batch_prediction(**context):
    """Score all subscribers using the production model."""
    import sys
    sys.path.insert(0, "/app")

    import pandas as pd
    import joblib
    from src.api.model_loader import ModelLoader

    model_art = joblib.load("data/models/churn_model_xgboost.pkl")
    df = pd.read_parquet("data/processed/features_full.parquet")

    from src.models.geo_risk_map import run_batch_prediction as score
    df_scored = score(df, model_art)

    df_scored[["subscriber_id", "churn_probability", "churn_predicted"]].to_parquet(
        "data/processed/predictions_latest.parquet", index=False
    )

    high_risk = (df_scored["churn_probability"] >= 0.5).sum()
    context["task_instance"].xcom_push(key="high_risk_count", value=int(high_risk))


def run_geo_risk_refresh(**context):
    """Refresh the H3 hex risk grid."""
    import subprocess
    result = subprocess.run(
        ["python", "src/models/geo_risk_map.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if result.returncode != 0:
        raise ValueError(f"Geo risk refresh failed:\n{result.stderr}")


def check_data_drift(**context):
    """
    Compute Population Stability Index (PSI) for key features.
    Returns 'retrain' branch if PSI > threshold, else 'skip_retrain'.
    """
    import numpy as np
    import pandas as pd
    import yaml

    config = yaml.safe_load(open("configs/config.yaml"))
    psi_threshold = config["monitoring"]["psi_threshold"]
    drift_features = config["monitoring"]["drift_check_features"]

    def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index — measures feature distribution shift."""
        eps = 1e-10
        bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)

        exp_pct = exp_counts / (len(expected) + eps)
        act_pct = act_counts / (len(actual) + eps)

        psi = np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps)))
        return float(psi)

    # Load reference (training) and current data
    ref = pd.read_parquet("data/processed/features_train.parquet")

    current_path = Path("data/processed/features_full.parquet")
    current = pd.read_parquet(current_path) if current_path.exists() else ref

    max_psi = 0.0
    drifted_features = []
    for feat in drift_features:
        if feat in ref.columns and feat in current.columns:
            psi_val = compute_psi(ref[feat].dropna().values, current[feat].dropna().values)
            if psi_val > psi_threshold:
                drifted_features.append((feat, round(psi_val, 4)))
            max_psi = max(max_psi, psi_val)

    context["task_instance"].xcom_push(key="max_psi", value=max_psi)
    context["task_instance"].xcom_push(key="drifted_features", value=drifted_features)

    if drifted_features:
        return "trigger_retrain"
    return "skip_retrain"


def trigger_model_retrain(**context):
    """Retrain the model when significant data drift is detected."""
    import subprocess

    drifted = context["task_instance"].xcom_pull(
        task_ids="drift_detection", key="drifted_features"
    )
    print(f"Retraining due to drift in features: {drifted}")

    result = subprocess.run(
        ["python", "src/models/train.py", "--model", "xgboost"],
        capture_output=True, text=True, cwd="/app"
    )
    if result.returncode != 0:
        raise ValueError(f"Retrain failed:\n{result.stderr}")


def send_pipeline_notification(**context):
    """Send pipeline summary notification (Slack / email)."""
    ti = context["task_instance"]
    high_risk = ti.xcom_pull(task_ids="batch_prediction", key="high_risk_count") or 0
    max_psi = ti.xcom_pull(task_ids="drift_detection", key="max_psi") or 0.0
    drifted = ti.xcom_pull(task_ids="drift_detection", key="drifted_features") or []
    run_date = context["ds"]

    message = (
        f"📡 Churn Pipeline Complete — {run_date}\n"
        f"  High-risk subscribers: {high_risk:,}\n"
        f"  Max PSI: {max_psi:.3f}\n"
        f"  Drifted features: {[f[0] for f in drifted] or 'None'}\n"
    )
    print(message)

    # ── Uncomment to send to Slack ──────────────────────────────────────────
    # import requests
    # webhook_url = Variable.get("SLACK_WEBHOOK_URL")
    # requests.post(webhook_url, json={"text": message})


# ── Task definitions ───────────────────────────────────────────────────────────

t_start = EmptyOperator(task_id="pipeline_start", dag=dag)

t_validate = PythonOperator(
    task_id="data_validation",
    python_callable=run_data_validation,
    dag=dag,
)

t_features = PythonOperator(
    task_id="feature_engineering",
    python_callable=run_feature_engineering,
    dag=dag,
)

t_predict = PythonOperator(
    task_id="batch_prediction",
    python_callable=run_batch_prediction,
    dag=dag,
)

t_geo = PythonOperator(
    task_id="geo_risk_refresh",
    python_callable=run_geo_risk_refresh,
    dag=dag,
)

t_drift = BranchPythonOperator(
    task_id="drift_detection",
    python_callable=check_data_drift,
    dag=dag,
)

t_retrain = PythonOperator(
    task_id="trigger_retrain",
    python_callable=trigger_model_retrain,
    dag=dag,
)

t_skip = EmptyOperator(task_id="skip_retrain", dag=dag)

t_notify = PythonOperator(
    task_id="send_notification",
    python_callable=send_pipeline_notification,
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)

t_end = EmptyOperator(task_id="pipeline_end", dag=dag)

# ── DAG topology ───────────────────────────────────────────────────────────────
#
#  start → validate → features → predict → geo_refresh → drift?
#                                                         ├─ retrain ─┐
#                                                         └─ skip    ─┤
#                                                                      └→ notify → end
#
t_start >> t_validate >> t_features >> t_predict >> t_geo >> t_drift
t_drift >> t_retrain >> t_notify
t_drift >> t_skip >> t_notify
t_notify >> t_end
