"""
src/features/subscriber_features.py
─────────────────────────────────────
Feature engineering for subscriber-level CRM and usage attributes.
All transformations are sklearn Pipeline-compatible.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class SubscriberFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features from raw subscriber data.
    Fit on train set, applied to test/production data.
    """

    def __init__(self):
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Tenure features ──────────────────────────────────────────────────
        df["is_new_subscriber"] = (df["tenure_months"] <= 3).astype(int)
        df["is_long_term"] = (df["tenure_months"] >= 24).astype(int)
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 3, 12, 24, 48, 200],
            labels=["0-3m", "3-12m", "12-24m", "24-48m", "48m+"],
        ).astype(str)

        # ── Charge features ──────────────────────────────────────────────────
        df["charge_per_month_tenure"] = (
            df["total_charges"] / df["tenure_months"].clip(1)
        ).round(2)
        df["monthly_charge_bucket"] = pd.cut(
            df["monthly_charges"],
            bins=[0, 30, 50, 75, 100, 200],
            labels=["<30", "30-50", "50-75", "75-100", "100+"],
        ).astype(str)

        # ── Usage intensity ratios ────────────────────────────────────────────
        df["data_charge_ratio"] = (
            df["data_usage_gb"] / df["monthly_charges"].clip(1)
        ).round(4)
        df["call_intensity"] = (
            df["call_minutes_monthly"] / df["tenure_months"].clip(1)
        ).round(2)

        # ── Contract risk encoding ────────────────────────────────────────────
        contract_risk = {
            "month-to-month": 3,
            "one-year": 2,
            "two-year": 1,
        }
        df["contract_risk_score"] = df["contract_type"].map(contract_risk).fillna(2)

        # ── Tech support engagement ───────────────────────────────────────────
        df["is_high_support"] = (df["tech_support_calls"] >= 3).astype(int)
        df["support_per_month"] = (
            df["tech_support_calls"] / df["tenure_months"].clip(1)
        ).round(4)

        logger.debug(f"Subscriber features added: {df.shape[1] - X.shape[1]} new columns")
        return df


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates derived features from network quality KPIs."""

    # RSRQ quality thresholds (3GPP standard)
    RSRQ_EXCELLENT = -9
    RSRQ_GOOD = -12
    RSRQ_POOR = -15

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # ── Signal quality categorisation ────────────────────────────────────
        df["rsrq_quality"] = pd.cut(
            df["rsrq_avg"],
            bins=[-np.inf, self.RSRQ_POOR, self.RSRQ_GOOD, self.RSRQ_EXCELLENT, np.inf],
            labels=["poor", "fair", "good", "excellent"],
        ).astype(str)

        df["rsrp_quality"] = pd.cut(
            df["rsrp_avg"],
            bins=[-np.inf, -110, -100, -90, -80, np.inf],
            labels=["very_poor", "poor", "fair", "good", "excellent"],
        ).astype(str)

        # ── Network quality composite score (0–100, higher = better) ─────────
        # Normalise each KPI and combine
        rsrq_norm = (df["rsrq_avg"].clip(-20, -3) - (-20)) / (-3 - (-20))  # 0 to 1
        rsrp_norm = (df["rsrp_avg"].clip(-130, -60) - (-130)) / (-60 - (-130))
        throughput_norm = np.log1p(df["dl_throughput_mbps"]) / np.log1p(150)

        df["network_quality_score"] = (
            (rsrq_norm * 0.40 + rsrp_norm * 0.35 + throughput_norm * 0.25) * 100
        ).round(1)

        # ── Drop call features ────────────────────────────────────────────────
        df["is_high_drop_rate"] = (df["call_drop_rate_pct"] > 5.0).astype(int)
        df["drop_rate_bucket"] = pd.cut(
            df["call_drop_rate_pct"],
            bins=[0, 1, 3, 5, 10, 100],
            labels=["<1%", "1-3%", "3-5%", "5-10%", "10%+"],
        ).astype(str)

        # ── Outage severity ───────────────────────────────────────────────────
        df["is_high_outage"] = (df["outage_minutes_monthly"] > 60).astype(int)
        df["outage_bucket"] = pd.cut(
            df["outage_minutes_monthly"],
            bins=[0, 15, 30, 60, 120, 1000],
            labels=["<15min", "15-30min", "30-60min", "1-2hr", "2hr+"],
        ).astype(str)

        # ── Combined network frustration index ───────────────────────────────
        df["network_frustration_index"] = (
            df["call_drops_monthly"] * 2.0
            + df["outage_minutes_monthly"] / 30.0
            + df["is_high_drop_rate"] * 3.0
            + (100 - df["network_quality_score"]) / 20.0
        ).round(2)

        logger.debug(f"Network features added: {df.shape[1] - X.shape[1]} new columns")
        return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns for tree models."""
    cat_cols = [
        "contract_type", "payment_method", "internet_service",
        "rsrq_quality", "rsrp_quality", "drop_rate_bucket",
        "outage_bucket", "tenure_bucket", "monthly_charge_bucket",
    ]
    existing_cats = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=False, dtype=int)
    return df


FEATURE_COLUMNS = [
    # Subscriber
    "tenure_months", "monthly_charges", "total_charges",
    "data_usage_gb", "call_minutes_monthly", "sms_monthly",
    "senior_citizen", "phone_service", "multiple_lines",
    "international_calls", "tech_support_calls",
    # Engineered subscriber
    "is_new_subscriber", "is_long_term", "charge_per_month_tenure",
    "data_charge_ratio", "call_intensity", "contract_risk_score",
    "is_high_support", "support_per_month",
    # Network
    "rsrp_avg", "rsrq_avg", "dl_throughput_mbps",
    "call_drop_rate_pct", "call_drops_monthly", "outage_minutes_monthly",
    # Engineered network
    "network_quality_score", "network_frustration_index",
    "is_high_drop_rate", "is_high_outage",
    # Geospatial
    "dist_to_nearest_tower_km", "towers_within_1km", "towers_within_2km",
    "cell_subscriber_count", "cell_avg_rsrq", "cell_avg_throughput",
    "cell_avg_outage", "cell_avg_monthly_charges",
]
