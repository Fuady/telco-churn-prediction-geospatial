"""
src/models/evaluate.py
───────────────────────
Standalone model evaluation script. Run after training to get full metrics,
SHAP analysis, threshold analysis and charts — all saved to data/models/.

Usage:
    python src/models/evaluate.py
    python src/models/evaluate.py --model_path data/models/churn_model_xgboost.pkl
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import yaml
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss, classification_report,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
    threshold: float,
    output_dir: Path,
) -> dict:
    """Run full evaluation suite and save charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test[feature_cols].fillna(-999))[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "brier":     round(brier_score_loss(y_test, y_prob), 4),
        "threshold": threshold,
        "n_test":    len(y_test),
        "churn_rate_actual":    round(y_test.mean(), 4),
        "churn_rate_predicted": round(y_pred.mean(), 4),
    }

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION REPORT")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # ── Confusion matrix chart ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, norm, title in [
        (axes[0], None, "Confusion Matrix"),
        (axes[1], "true", "Normalised Confusion Matrix"),
    ]:
        cm_plot = confusion_matrix(y_test, y_pred, normalize=norm)
        fmt = ".2f" if norm else "d"
        import seaborn as sns
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=["Retained", "Churned"],
                    yticklabels=["Retained", "Churned"],
                    cbar=False, linewidths=0.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {cm_path}")

    # ── ROC + PR curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="#e74c3c", lw=2.5,
                 label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve"); axes[0].legend()

    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec_c, prec_c, color="#185FA5", lw=2.5,
                 label=f"AP = {metrics['pr_auc']:.3f}")
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1,
                   label=f"Baseline = {y_test.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()
    roc_path = output_dir / "roc_pr_curves.png"
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {roc_path}")

    # ── Threshold tuning chart ─────────────────────────────────────────────────
    thresh_range = np.arange(0.10, 0.80, 0.01)
    f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]
    prec_scores = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]
    rec_scores  = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresh_range]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_range, f1_scores,   color="#185FA5", lw=2, label="F1")
    ax.plot(thresh_range, prec_scores, color="#e74c3c", lw=2, label="Precision")
    ax.plot(thresh_range, rec_scores,  color="#2ecc71", lw=2, label="Recall")
    ax.axvline(threshold, color="k", ls="--", lw=1.5, label=f"Chosen ({threshold:.2f})")
    ax.set(xlabel="Threshold", ylabel="Score", title="Metrics vs Classification Threshold")
    ax.legend(); plt.tight_layout()
    thresh_path = output_dir / "threshold_analysis.png"
    plt.savefig(thresh_path, dpi=120, bbox_inches="tight"); plt.close()
    logger.info(f"Saved {thresh_path}")

    # ── Cumulative gain chart ──────────────────────────────────────────────────
    rank_df = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob})
    rank_df = rank_df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    total = rank_df["y_true"].sum()
    rank_df["cum_gain"] = rank_df["y_true"].cumsum() / total
    rank_df["pct_pop"]  = (rank_df.index + 1) / len(rank_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rank_df["pct_pop"] * 100, rank_df["cum_gain"] * 100,
            color="#e74c3c", lw=2.5, label="XGBoost")
    ax.plot([0, 100], [0, 100], color="#95a5a6", ls=":", lw=1.5, label="Random")
    ax.set(xlabel="% Subscribers Contacted (ranked by probability)",
           ylabel="% Churners Captured", title="Cumulative Gain Chart")
    ax.legend(); plt.tight_layout()
    gain_path = output_dir / "cumulative_gain.png"
    plt.savefig(gain_path, dpi=120, bbox_inches="tight"); plt.close()
    logger.info(f"Saved {gain_path}")

    # ── Save metrics as CSV ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.success(f"Metrics saved → {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained churn model")
    parser.add_argument("--model_path", type=str,
                        default="data/models/churn_model_xgboost.pkl")
    parser.add_argument("--test_data", type=str,
                        default="data/processed/features_test.parquet")
    parser.add_argument("--output_dir", type=str, default="data/models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run: python src/models/train.py")
        sys.exit(1)

    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}. Run: python src/features/feature_pipeline.py")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = artifact.get("threshold", 0.45)

    logger.info(f"Loading test data: {test_path}")
    df_test = pd.read_parquet(test_path)

    cfg = load_config(args.config)
    target = cfg["features"]["target_column"]

    y_test = df_test[target]
    X_test = df_test

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        threshold=threshold,
        output_dir=Path(args.output_dir),
    )

    logger.success("Evaluation complete.")


if __name__ == "__main__":
    main()
