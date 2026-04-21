"""
tests/test_model.py
────────────────────
Unit and integration tests for the model training and prediction modules.
Run: pytest tests/test_model.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_binary_dataset(n: int = 500, n_features: int = 10, seed: int = 42):
    """Create a minimal binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Make target correlated with first feature
    logit = 0.8 * X["feat_0"] - 0.4 * X["feat_1"] + rng.normal(0, 0.5, n)
    y = pd.Series((1 / (1 + np.exp(-logit)) > 0.5).astype(int), name="churned")
    return X, y


class TestXGBoostTraining:

    def test_model_fits_without_error(self):
        import xgboost as xgb
        X, y = make_binary_dataset()
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        assert hasattr(model, "feature_importances_")

    def test_predict_proba_shape(self):
        import xgboost as xgb
        X, y = make_binary_dataset()
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_values_in_range(self):
        import xgboost as xgb
        X, y = make_binary_dataset()
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_roc_auc_above_random(self):
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = make_binary_dataset(n=1000)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        assert auc > 0.60, f"AUC {auc:.3f} is barely above random"

    def test_feature_importances_non_negative(self):
        import xgboost as xgb
        X, y = make_binary_dataset()
        model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X, y)
        assert (model.feature_importances_ >= 0).all()


class TestSMOTE:

    def test_smote_increases_minority_class(self):
        from imblearn.over_sampling import SMOTE
        X, _ = make_binary_dataset(n=500)
        # Heavily imbalanced: 90% negative
        y = pd.Series(np.where(np.arange(500) < 450, 0, 1))
        original_positive = y.sum()

        sm = SMOTE(sampling_strategy=0.5, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        assert y_res.sum() > original_positive
        assert y_res.mean() > y.mean()

    def test_smote_preserves_features(self):
        from imblearn.over_sampling import SMOTE
        X, y = make_binary_dataset(n=300)
        y = pd.Series(np.where(y == 1, 1, 0))

        sm = SMOTE(sampling_strategy=0.5, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        assert X_res.shape[1] == X.shape[1]
        assert list(X_res.columns) == list(X.columns)


class TestThresholdTuning:

    def test_threshold_tuning_returns_valid_threshold(self):
        from src.models.train import tune_threshold
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        thresh = tune_threshold(y_true, y_prob)
        assert 0.10 <= thresh <= 0.80

    def test_tune_threshold_improves_f1(self):
        from src.models.train import tune_threshold
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 500)
        # Assign probabilities correlated with labels
        y_prob = np.where(y_true == 1,
                          rng.uniform(0.5, 1.0, 500),
                          rng.uniform(0.0, 0.6, 500))

        best_thresh = tune_threshold(y_true, y_prob)
        default_f1 = f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)
        tuned_f1   = f1_score(y_true, (y_prob >= best_thresh).astype(int), zero_division=0)
        # Tuned threshold should be at least as good as default
        assert tuned_f1 >= default_f1 - 0.01


class TestEvaluateModule:

    def test_compute_metrics_keys(self):
        from src.models.train import compute_metrics
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        for key in ["roc_auc", "pr_auc", "f1", "precision", "recall", "threshold"]:
            assert key in metrics

    def test_metric_values_in_range(self):
        from src.models.train import compute_metrics
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        metrics = compute_metrics(y_true, y_prob, threshold=0.5)
        for key in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of range"


class TestGeoRiskMap:

    def test_assign_risk_tier(self):
        from src.models.geo_risk_map import assign_risk_tier
        tiers = {
            "LOW":      (0.00, 0.25),
            "MEDIUM":   (0.25, 0.50),
            "HIGH":     (0.50, 0.70),
            "CRITICAL": (0.70, 1.01),
        }
        assert assign_risk_tier(0.10, tiers) == "LOW"
        assert assign_risk_tier(0.35, tiers) == "MEDIUM"
        assert assign_risk_tier(0.60, tiers) == "HIGH"
        assert assign_risk_tier(0.85, tiers) == "CRITICAL"

    def test_aggregate_to_h3_filters_thin_cells(self):
        from src.models.geo_risk_map import aggregate_to_h3
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "h3_r8": ["cell_A"] * 100 + ["cell_B"] * 90 + ["cell_C"] * 10,
            "churn_predicted": rng.integers(0, 2, n),
            "churn_probability": rng.uniform(0, 1, n),
            "rsrq_avg": rng.uniform(-20, -3, n),
            "dl_throughput_mbps": rng.uniform(1, 100, n),
            "monthly_charges": rng.uniform(20, 100, n),
            "outage_minutes_monthly": rng.uniform(0, 100, n),
            "contract_risk_score": rng.choice([1, 2, 3], n),
        })
        agg = aggregate_to_h3(df, "h3_r8", min_subscribers=15)
        # cell_C has only 10 rows → should be filtered out
        assert len(agg) == 2
        assert "cell_C" not in agg["h3_r8"].values

    def test_revenue_at_risk_calculation(self):
        from src.models.geo_risk_map import aggregate_to_h3
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "h3_r8": ["cell_X"] * n,
            "churn_predicted": [1] * 10 + [0] * 40,
            "churn_probability": [0.9] * 10 + [0.1] * 40,
            "rsrq_avg": [-11.0] * n,
            "dl_throughput_mbps": [20.0] * n,
            "monthly_charges": [100.0] * n,
            "outage_minutes_monthly": [30.0] * n,
            "contract_risk_score": [2] * n,
        })
        agg = aggregate_to_h3(df, "h3_r8", min_subscribers=5)
        assert "estimated_revenue_at_risk" in agg.columns
        assert agg["estimated_revenue_at_risk"].iloc[0] > 0
