"""
src/models/train.py
────────────────────
Model training pipeline with MLflow experiment tracking.

Features:
  - Trains XGBoost and LightGBM with class imbalance handling (SMOTE)
  - Optuna hyperparameter optimisation (optional --tune flag)
  - MLflow logging: params, metrics, artifacts, model
  - Threshold tuning for optimal F1 / business recall target
  - Registers best model to MLflow Model Registry

Usage:
    python src/models/train.py
    python src/models/train.py --tune            # run Optuna HPO
    python src/models/train.py --model lightgbm
    python src/models/train.py --config configs/model_params.yaml
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import yaml
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_dir: Path, target: str, id_cols: list[str]):
    """Load train/test sets and split X/y."""
    df_train = pd.read_parquet(processed_dir / "features_train.parquet")
    df_test = pd.read_parquet(processed_dir / "features_test.parquet")

    feature_path = processed_dir / "feature_columns.txt"
    if feature_path.exists():
        feature_cols = feature_path.read_text().strip().split("\n")
    else:
        exclude = id_cols + [target, "churn_probability_true", "snapshot_date"]
        feature_cols = [c for c in df_train.columns if c not in exclude
                        and df_train[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]]

    # Filter to numeric columns only (safety check)
    feature_cols = [c for c in feature_cols if c in df_train.columns
                    and pd.api.types.is_numeric_dtype(df_train[c])]

    X_train = df_train[feature_cols].fillna(-999)
    y_train = df_train[target]
    X_test = df_test[feature_cols].fillna(-999)
    y_test = df_test[target]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train: {X_train.shape} | Churn rate: {y_train.mean():.2%}")
    logger.info(f"Test:  {X_test.shape} | Churn rate: {y_test.mean():.2%}")

    return X_train, y_train, X_test, y_test, feature_cols


def apply_smote(X: pd.DataFrame, y: pd.Series, params: dict) -> tuple:
    """Oversample minority class using SMOTE."""
    logger.info("Applying SMOTE to handle class imbalance...")
    sm = SMOTE(
        sampling_strategy=params.get("sampling_strategy", 0.5),
        k_neighbors=params.get("k_neighbors", 5),
        random_state=params.get("random_state", 42),
    )
    X_res, y_res = sm.fit_resample(X, y)
    logger.info(f"After SMOTE — shape: {X_res.shape} | churn rate: {y_res.mean():.2%}")
    return X_res, y_res


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on the positive class."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.20, 0.70, 0.01):
        preds = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    logger.info(f"Best threshold: {best_thresh:.2f} → F1={best_f1:.4f}")
    return best_thresh


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def plot_feature_importance(model, feature_cols: list, output_path: Path, top_n: int = 30):
    """Save SHAP feature importance plot."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.nlargest(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        imp.sort_values().plot.barh(ax=ax, color="#5E5BE0")
        ax.set_title(f"Top {top_n} Feature Importances (Gain)")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved → {output_path}")


def plot_shap_summary(model, X_sample: pd.DataFrame, output_path: Path):
    """Save SHAP beeswarm summary plot."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # class 1 for binary

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved → {output_path}")
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")


def train_xgboost(X_train, y_train, X_val, y_val, params: dict):
    """Train XGBoost model."""
    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 5),
        gamma=params.get("gamma", 0.1),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 4.5),
        eval_metric="aucpr",
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        random_state=params.get("random_state", 42),
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict):
    """Train LightGBM model."""
    callbacks = [lgb.early_stopping(params.get("early_stopping_rounds", 50), verbose=False),
                 lgb.log_evaluation(period=-1)]
    model = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_samples=params.get("min_child_samples", 20),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        is_unbalance=params.get("is_unbalance", True),
        random_state=params.get("random_state", 42),
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--params", type=str, default="configs/model_params.yaml")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter optimisation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_params = load_config(args.params)

    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path("data/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    target = cfg["features"]["target_column"]
    id_cols = ["subscriber_id", "snapshot_date", "h3_r7", "h3_r8",
               "nearest_tower_radio", "churn_probability_true"]

    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        processed_dir, target, id_cols
    )

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train, model_params["smote"])

    # Validation split from resampled training data
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_res, y_train_res, test_size=0.15, random_state=42, stratify=y_train_res
    )

    # ── Optional: Optuna HPO ─────────────────────────────────────────────────
    if args.tune:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("Running Optuna hyperparameter optimisation...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": 4.5,
                "random_state": 42,
                "early_stopping_rounds": 30,
            }
            m = train_xgboost(X_tr, y_tr, X_val, y_val, params)
            prob = m.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, timeout=600)
        best = study.best_params
        logger.info(f"Best params: {best}")
        model_params["xgboost"].update(best)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["model"]["experiment_name"])

    with mlflow.start_run(run_name=f"{args.model}_churn") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        params = model_params[args.model]
        mlflow.log_params(params)
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("smote_applied", True)

        # ── Train ─────────────────────────────────────────────────────────────
        logger.info(f"Training {args.model}...")
        if args.model == "xgboost":
            model = train_xgboost(X_tr, y_tr, X_val, y_val, params)
        else:
            model = train_lightgbm(X_tr, y_tr, X_val, y_val, params)

        # ── Evaluate on test set ──────────────────────────────────────────────
        y_prob_test = model.predict_proba(X_test)[:, 1]
        threshold = tune_threshold(y_test.values, y_prob_test)
        metrics = compute_metrics(y_test.values, y_prob_test, threshold)

        logger.info("Test set metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
            mlflow.log_metric(k, round(float(v), 4))

        # ── Save artifacts ────────────────────────────────────────────────────
        # Feature importance plot
        fi_path = artifacts_dir / "feature_importance.png"
        plot_feature_importance(model, feature_cols, fi_path)
        mlflow.log_artifact(str(fi_path))

        # SHAP summary (sample 2000 rows for speed)
        shap_path = artifacts_dir / "shap_summary.png"
        sample_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
        plot_shap_summary(model, X_test.iloc[sample_idx], shap_path)
        mlflow.log_artifact(str(shap_path))

        # Save model locally
        model_path = artifacts_dir / f"churn_model_{args.model}.pkl"
        joblib.dump({"model": model, "feature_cols": feature_cols, "threshold": threshold}, model_path)
        mlflow.log_artifact(str(model_path))

        # Save feature list
        feat_path = artifacts_dir / "feature_columns.txt"
        feat_path.write_text("\n".join(feature_cols))
        mlflow.log_artifact(str(feat_path))

        # Log model to MLflow
        if args.model == "xgboost":
            mlflow.xgboost.log_model(
                model, "model",
                registered_model_name=cfg["model"]["registered_model_name"],
            )
        else:
            mlflow.lightgbm.log_model(
                model, "model",
                registered_model_name=cfg["model"]["registered_model_name"],
            )

        logger.success(f"Model registered as '{cfg['model']['registered_model_name']}'")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model        : {args.model}")
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC       : {metrics['pr_auc']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  Threshold    : {metrics['threshold']:.2f}")
    print(f"\n  MLflow UI    : {cfg['mlflow']['tracking_uri']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
