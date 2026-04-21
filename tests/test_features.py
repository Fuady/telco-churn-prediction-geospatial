"""
tests/test_features.py
───────────────────────
Unit tests for the feature engineering pipeline.
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.subscriber_features import (
    SubscriberFeatureEngineer,
    NetworkFeatureEngineer,
    encode_categoricals,
)


def make_sample_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal sample subscriber DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "subscriber_id": [f"SUB_{i}" for i in range(n)],
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "total_charges": rng.uniform(100, 5000, n),
        "data_usage_gb": rng.uniform(0.1, 50, n),
        "call_minutes_monthly": rng.uniform(0, 1000, n),
        "sms_monthly": rng.uniform(0, 300, n),
        "tech_support_calls": rng.integers(0, 10, n),
        "senior_citizen": rng.integers(0, 2, n),
        "phone_service": rng.integers(0, 2, n),
        "multiple_lines": rng.integers(0, 2, n),
        "international_calls": rng.integers(0, 2, n),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n),
        "payment_method": rng.choice(["electronic_check", "credit_card"], n),
        "internet_service": rng.choice(["fiber_optic", "DSL", "none"], n),
        "rsrp_avg": rng.uniform(-130, -60, n),
        "rsrq_avg": rng.uniform(-20, -3, n),
        "dl_throughput_mbps": rng.uniform(0.5, 100, n),
        "call_drop_rate_pct": rng.uniform(0, 15, n),
        "call_drops_monthly": rng.integers(0, 20, n),
        "outage_minutes_monthly": rng.integers(0, 200, n),
        "latitude": rng.uniform(-7.0, -5.5, n),
        "longitude": rng.uniform(106.4, 107.5, n),
        "churned": rng.integers(0, 2, n),
    })


class TestSubscriberFeatureEngineer:

    def test_transform_runs(self):
        df = make_sample_df()
        eng = SubscriberFeatureEngineer()
        eng.fit(df)
        result = eng.transform(df)
        assert len(result) == len(df)

    def test_new_columns_created(self):
        df = make_sample_df()
        eng = SubscriberFeatureEngineer().fit(df)
        result = eng.transform(df)
        for col in ["is_new_subscriber", "is_long_term", "contract_risk_score",
                    "data_charge_ratio", "call_intensity", "tenure_bucket"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_new_subscriber_correct(self):
        df = make_sample_df()
        df["tenure_months"] = [2, 5, 12, 1][:len(df)] + [12] * (len(df) - 4)
        result = SubscriberFeatureEngineer().fit(df).transform(df)
        # tenure <= 3 should be new
        assert result.loc[df["tenure_months"] <= 3, "is_new_subscriber"].all()
        assert not result.loc[df["tenure_months"] > 3, "is_new_subscriber"].any()

    def test_contract_risk_encoding(self):
        df = make_sample_df(3)
        df["contract_type"] = ["month-to-month", "one-year", "two-year"]
        result = SubscriberFeatureEngineer().fit(df).transform(df)
        scores = result["contract_risk_score"].tolist()
        assert scores[0] == 3  # month-to-month highest risk
        assert scores[1] == 2
        assert scores[2] == 1

    def test_no_nulls_in_engineered_features(self):
        df = make_sample_df()
        result = SubscriberFeatureEngineer().fit(df).transform(df)
        numeric_new = ["is_new_subscriber", "contract_risk_score",
                       "data_charge_ratio", "call_intensity"]
        for col in numeric_new:
            assert result[col].isnull().sum() == 0, f"Nulls in {col}"


class TestNetworkFeatureEngineer:

    def test_transform_runs(self):
        df = make_sample_df()
        eng = NetworkFeatureEngineer().fit(df)
        result = eng.transform(df)
        assert len(result) == len(df)

    def test_network_quality_score_range(self):
        df = make_sample_df()
        result = NetworkFeatureEngineer().fit(df).transform(df)
        assert "network_quality_score" in result.columns
        assert result["network_quality_score"].between(0, 100).all(), \
            "network_quality_score must be in [0, 100]"

    def test_frustration_index_non_negative(self):
        df = make_sample_df()
        result = NetworkFeatureEngineer().fit(df).transform(df)
        assert (result["network_frustration_index"] >= 0).all()

    def test_rsrq_quality_buckets(self):
        df = make_sample_df(3)
        df["rsrq_avg"] = [-8.0, -13.0, -18.0]  # excellent, fair, poor
        result = NetworkFeatureEngineer().fit(df).transform(df)
        assert result["rsrq_quality"].iloc[0] == "excellent"
        assert result["rsrq_quality"].iloc[2] == "poor"


class TestEncodeCategoricals:

    def test_contract_type_encoded(self):
        df = make_sample_df()
        df = SubscriberFeatureEngineer().fit(df).transform(df)
        df = NetworkFeatureEngineer().fit(df).transform(df)
        result = encode_categoricals(df)
        assert "contract_type" not in result.columns
        assert any(c.startswith("contract_type_") for c in result.columns)

    def test_all_numeric_after_encoding(self):
        df = make_sample_df()
        df = SubscriberFeatureEngineer().fit(df).transform(df)
        df = NetworkFeatureEngineer().fit(df).transform(df)
        result = encode_categoricals(df)
        non_numeric = [
            c for c in result.columns
            if not pd.api.types.is_numeric_dtype(result[c])
            and c not in ["subscriber_id", "snapshot_date",
                          "h3_r7", "h3_r8", "nearest_tower_radio"]
        ]
        assert len(non_numeric) == 0, f"Non-numeric columns remaining: {non_numeric}"
