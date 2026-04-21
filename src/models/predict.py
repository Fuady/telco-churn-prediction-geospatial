"""
src/models/predict.py
──────────────────────
Batch prediction script for scoring new subscriber data.
Designed for scheduled production use (e.g. weekly Airflow run).

Workflow:
  1. Load the production model from MLflow or local pickle
  2. Load and validate new subscriber data
  3. Run feature engineering pipeline
  4. Score all subscribers
  5. Export predictions + risk tiers to Parquet and CSV
  6. Generate a summary report

Usage:
    # Score from raw data (runs feature engineering automatically)
    python src/models/predict.py --input data/raw/subscribers.parquet

    # Score from already-processed feature data
    python src/models/predict.py --input data/processed/features_full.parquet --skip_features

    # With custom output path
    python src/models/predict.py --input data/raw/ --output data/processed/predictions.parquet
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_engineering.data_validation import validate_subscribers


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def assign_risk_tier(prob: float, tiers: dict) -> str:
    for tier, (lo, hi) in tiers.items():
        if lo <= prob < hi:
            return tier
    return "CRITICAL"


def load_model(model_path: str, config: dict):
    """Load model from local pickle, with MLflow fallback."""
    p = Path(model_path)
    if p.exists():
        logger.info(f"Loading model from {p}")
        return joblib.load(p)

    # Try MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        name = config["model"]["registered_model_name"]
        stage = config["api"]["model_stage"]
        logger.info(f"Loading model from MLflow: {name}@{stage}")
        mlflow_model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
        feat_path = Path("data/models/feature_columns.txt")
        feature_cols = feat_path.read_text().strip().split("\n") if feat_path.exists() else []
        return {"model": mlflow_model, "feature_cols": feature_cols, "threshold": 0.45}
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        logger.error("Run: python src/models/train.py")
        sys.exit(1)


def run_features_if_needed(
    input_path: Path,
    config: dict,
    skip_features: bool,
) -> pd.DataFrame:
    """Load data, running feature engineering if input is raw."""
    if skip_features or "features_" in input_path.name:
        logger.info(f"Loading pre-processed features: {input_path}")
        return pd.read_parquet(input_path)

    # Check if raw subscriber file
    logger.info("Input appears to be raw data — running feature engineering...")
    from src.features.feature_pipeline import run_pipeline
    run_pipeline(
        config=config,
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=Path("data/processed"),
    )
    return pd.read_parquet("data/processed/features_full.parquet")


def batch_predict(
    df: pd.DataFrame,
    model_artifact: dict,
    tiers: dict,
) -> pd.DataFrame:
    """Score all subscribers and add prediction columns."""
    model       = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    threshold   = model_artifact.get("threshold", 0.45)

    # Align features
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} features missing — filling with 0: {missing[:5]}...")
        for c in missing:
            df[c] = 0

    X = df[feature_cols].fillna(-999)

    logger.info(f"Scoring {len(X):,} subscribers...")

    # Support both sklearn and MLflow pyfunc
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X).values

    y_pred = (y_prob >= threshold).astype(int)
    y_tier = [assign_risk_tier(p, tiers) for p in y_prob]

    df = df.copy()
    df["churn_probability"] = y_prob.round(4)
    df["churn_predicted"]   = y_pred
    df["risk_tier"]         = y_tier
    df["scored_at"]         = datetime.utcnow().isoformat()

    # Monthly revenue at risk
    if "monthly_charges" in df.columns:
        df["revenue_at_risk"] = (df["monthly_charges"] * df["churn_probability"]).round(2)

    return df


def generate_summary(df_scored: pd.DataFrame) -> None:
    """Print a business-friendly prediction summary."""
    total = len(df_scored)
    churn_count  = df_scored["churn_predicted"].sum()
    churn_rate   = churn_count / total

    print("\n" + "=" * 60)
    print("  BATCH PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Scored at       : {df_scored['scored_at'].iloc[0]}")
    print(f"  Total scored    : {total:,}")
    print(f"  Predicted churn : {churn_count:,} ({churn_rate:.2%})")

    if "risk_tier" in df_scored.columns:
        print(f"\n  Risk Tier Breakdown:")
        tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        tier_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        for tier in tier_order:
            count = (df_scored["risk_tier"] == tier).sum()
            pct   = count / total * 100
            icon  = tier_colors.get(tier, " ")
            print(f"    {icon} {tier:10s}: {count:6,} ({pct:.1f}%)")

    if "revenue_at_risk" in df_scored.columns:
        total_rev = df_scored["revenue_at_risk"].sum()
        print(f"\n  Total revenue at risk: ${total_rev:,.0f}/month")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Batch score subscribers for churn risk")
    parser.add_argument("--input", type=str, default="data/processed/features_full.parquet",
                        help="Path to input data (raw Parquet or processed features)")
    parser.add_argument("--output", type=str, default="data/processed/predictions_latest.parquet")
    parser.add_argument("--model_path", type=str,
                        default="data/models/churn_model_xgboost.pkl")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--skip_features", action="store_true",
                        help="Skip feature engineering (use if input is already processed)")
    parser.add_argument("--export_csv", action="store_true",
                        help="Also export predictions as CSV (in addition to Parquet)")
    args = parser.parse_args()

    config = load_config(args.config)

    # ── Load model ─────────────────────────────────────────────────────────────
    model_artifact = load_model(args.model_path, config)

    # ── Load data ──────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    df = run_features_if_needed(input_path, config, args.skip_features)
    logger.info(f"Loaded {len(df):,} subscribers")

    # ── Score ──────────────────────────────────────────────────────────────────
    tiers = {k: tuple(v) for k, v in config["geo_risk"]["risk_tiers"].items()}
    df_scored = batch_predict(df, model_artifact, tiers)

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save minimal prediction output (not all features — keeps file small)
    id_cols = ["subscriber_id"]
    pred_cols = ["churn_probability", "churn_predicted", "risk_tier",
                 "revenue_at_risk", "scored_at"]
    geo_cols  = ["latitude", "longitude", "h3_r8", "h3_r7"]

    output_cols = (
        [c for c in id_cols if c in df_scored.columns] +
        [c for c in pred_cols if c in df_scored.columns] +
        [c for c in geo_cols if c in df_scored.columns]
    )

    df_out = df_scored[output_cols]
    df_out.to_parquet(output_path, index=False)
    logger.success(f"Predictions saved → {output_path} ({len(df_out):,} rows)")

    if args.export_csv:
        csv_path = output_path.with_suffix(".csv")
        df_out.to_csv(csv_path, index=False)
        logger.success(f"CSV saved → {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    generate_summary(df_scored)


if __name__ == "__main__":
    main()
