"""
src/api/model_loader.py
────────────────────────
Loads the trained model artifact and runs inference.
Handles feature engineering for API inputs (mirrors the training pipeline).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.subscriber_features import (
    SubscriberFeatureEngineer,
    NetworkFeatureEngineer,
    encode_categoricals,
)

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

RISK_TIERS = {
    "LOW":      (0.00, 0.25),
    "MEDIUM":   (0.25, 0.50),
    "HIGH":     (0.50, 0.70),
    "CRITICAL": (0.70, 1.01),
}


def assign_risk_tier(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "CRITICAL"


class ModelLoader:
    """Singleton model loader — loads once on startup, reused for every request."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.model = None
        self.feature_cols: list[str] = []
        self.threshold: float = 0.45
        self.model_type: str = "unknown"
        self.model_version: Optional[str] = None
        self._sub_eng = SubscriberFeatureEngineer()
        self._net_eng = NetworkFeatureEngineer()
        self._sub_eng.fit(pd.DataFrame())  # stateless — safe to pre-fit
        self._net_eng.fit(pd.DataFrame())

    def load(self, model_path: Optional[str] = None) -> None:
        """Load model artifact from disk (or MLflow if configured)."""
        if model_path is None:
            # Try MLflow first, fall back to local pickle
            try:
                self._load_from_mlflow()
                return
            except Exception as e:
                logger.warning(f"MLflow load failed ({e}), trying local pickle...")

        # Load local pickle
        local_paths = [
            model_path or "",
            "data/models/churn_model_xgboost.pkl",
            "data/models/churn_model_lightgbm.pkl",
        ]
        for path in local_paths:
            p = Path(path)
            if p.exists():
                self._load_from_pickle(p)
                return

        logger.error(
            "No model found. Run: python src/models/train.py\n"
            "Then retry starting the API."
        )

    def _load_from_pickle(self, path: Path) -> None:
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = artifact.get("threshold", 0.45)
        self.model_type = type(self.model).__name__
        self.model_version = "local"
        logger.success(f"Model loaded from {path} ({self.model_type})")

    def _load_from_mlflow(self) -> None:
        import mlflow
        config = yaml.safe_load(open(self.config_path))
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        model_name = config["model"]["registered_model_name"]
        stage = config["api"]["model_stage"]

        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model from MLflow: {model_uri}")
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.model_version = stage

        # Load feature columns from artifact
        feat_path = Path("data/models/feature_columns.txt")
        if feat_path.exists():
            self.feature_cols = feat_path.read_text().strip().split("\n")

        threshold_path = Path("data/models/threshold.txt")
        if threshold_path.exists():
            self.threshold = float(threshold_path.read_text().strip())

        self.model_type = "MLflow/PyFunc"
        logger.success(f"Model loaded from MLflow ({model_name}@{stage})")

    def is_loaded(self) -> bool:
        return self.model is not None

    def _build_features(self, raw: dict) -> pd.DataFrame:
        """Apply the same feature engineering as training pipeline."""
        df = pd.DataFrame([raw])

        # Fill defaults for missing fields
        defaults = {
            "churned": 0, "churn_probability_true": 0,
            "h3_r7": "unknown", "h3_r8": "unknown",
            "dist_to_nearest_tower_km": 1.5,
            "towers_within_1km": 3,
            "towers_within_2km": 10,
            "nearest_tower_radio": "LTE",
            "cell_subscriber_count": 100,
            "cell_avg_rsrq": -11.0,
            "cell_avg_rsrp": -95.0,
            "cell_avg_throughput": 20.0,
            "cell_avg_outage": 30.0,
            "cell_avg_monthly_charges": 60.0,
            "cell_churn_rate": 0.15,
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v

        df = self._sub_eng.transform(df)
        df = self._net_eng.transform(df)
        df = encode_categoricals(df)

        # Align feature columns — fill any missing with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_cols].fillna(-999)

    def predict_single(self, raw: dict) -> dict:
        """Run inference on a single subscriber dict."""
        subscriber_id = raw.get("subscriber_id", "unknown")
        lat = raw.get("latitude", 0.0)
        lon = raw.get("longitude", 0.0)
        monthly_charges = raw.get("monthly_charges", 0.0)

        X = self._build_features(raw)

        # Get probability
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            # MLflow pyfunc
            prob = float(self.model.predict(X).iloc[0])

        label = int(prob >= self.threshold)
        tier = assign_risk_tier(prob)

        # H3 cell
        h3_cell = None
        if H3_AVAILABLE:
            try:
                import h3 as h3lib
                h3_cell = h3lib.geo_to_h3(lat, lon, 8)
            except Exception:
                pass

        # Top factors — use feature importances if available
        top_factors = self._get_top_factors(X)

        return {
            "subscriber_id": subscriber_id,
            "churn_probability": round(prob, 4),
            "churn_label": label,
            "risk_tier": tier,
            "h3_cell": h3_cell,
            "top_factors": top_factors,
            "monthly_revenue_at_risk": round(monthly_charges * prob, 2),
        }

    def predict_batch(self, raw_list: list[dict]) -> list[dict]:
        """Run inference on a list of subscriber dicts."""
        return [self.predict_single(r) for r in raw_list]

    def _get_top_factors(self, X: pd.DataFrame, top_n: int = 3) -> list[str]:
        """Return the top N feature names by importance for this prediction."""
        model = self.model
        # Handle MLflow wrapper
        if hasattr(model, "_model_impl"):
            model = model._model_impl

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=self.feature_cols)
            # Weight by absolute feature value for this subscriber
            abs_vals = X.iloc[0].abs()
            contribution = imp * abs_vals
            return contribution.nlargest(top_n).index.tolist()
        return []
