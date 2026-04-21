"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI prediction service.
Run: pytest tests/test_api.py -v

NOTE: These tests use TestClient (no server needed).
For live API tests: set TEST_API_URL=http://localhost:8000
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SAMPLE_PAYLOAD = {
    "subscriber_id": "TEST_001",
    "senior_citizen": 0,
    "phone_service": 1,
    "multiple_lines": 0,
    "internet_service": "fiber_optic",
    "international_calls": 0,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "tenure_months": 18,
    "monthly_charges": 65.0,
    "total_charges": 1170.0,
    "data_usage_gb": 12.5,
    "call_minutes_monthly": 300.0,
    "sms_monthly": 50.0,
    "tech_support_calls": 1,
    "rsrp_avg": -95.0,
    "rsrq_avg": -11.0,
    "dl_throughput_mbps": 20.0,
    "call_drop_rate_pct": 2.0,
    "call_drops_monthly": 1,
    "outage_minutes_monthly": 30.0,
    "latitude": -6.2088,
    "longitude": 106.8456,
}


def make_mock_model():
    """Create a mock model artifact for testing without a real trained model."""
    import numpy as np
    import pandas as pd

    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
    mock_model.feature_importances_ = np.ones(5)

    return {
        "model": mock_model,
        "feature_cols": [
            "tenure_months", "monthly_charges", "total_charges",
            "data_usage_gb", "rsrq_avg",
        ],
        "threshold": 0.45,
    }


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_valid_payload_parses(self):
        from src.api.schemas import SubscriberInput
        sub = SubscriberInput(**SAMPLE_PAYLOAD)
        assert sub.subscriber_id == "TEST_001"
        assert sub.churn_probability is None  # not a prediction field

    def test_invalid_contract_type_rejected(self):
        from src.api.schemas import SubscriberInput
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "contract_type": "weekly"}
        with pytest.raises(ValidationError):
            SubscriberInput(**bad)

    def test_invalid_internet_service_rejected(self):
        from src.api.schemas import SubscriberInput
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "internet_service": "satellite"}
        with pytest.raises(ValidationError):
            SubscriberInput(**bad)

    def test_negative_tenure_rejected(self):
        from src.api.schemas import SubscriberInput
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "tenure_months": -5}
        with pytest.raises(ValidationError):
            SubscriberInput(**bad)

    def test_latitude_out_of_range_rejected(self):
        from src.api.schemas import SubscriberInput
        from pydantic import ValidationError
        bad = {**SAMPLE_PAYLOAD, "latitude": 200.0}
        with pytest.raises(ValidationError):
            SubscriberInput(**bad)


class TestModelLoader:
    """Test ModelLoader without a real model file."""

    def test_is_loaded_false_initially(self):
        from src.api.model_loader import ModelLoader
        loader = ModelLoader()
        assert not loader.is_loaded()

    def test_predict_single_returns_correct_keys(self):
        from src.api.model_loader import ModelLoader
        import numpy as np

        loader = ModelLoader()
        artifact = make_mock_model()
        loader.model = artifact["model"]
        loader.feature_cols = artifact["feature_cols"]
        loader.threshold = artifact["threshold"]

        # Mock the feature engineering to return the right shape
        import pandas as pd
        mock_X = pd.DataFrame(np.zeros((1, 5)), columns=artifact["feature_cols"])

        with patch.object(loader, "_build_features", return_value=mock_X):
            result = loader.predict_single(SAMPLE_PAYLOAD)

        assert "churn_probability" in result
        assert "churn_label" in result
        assert "risk_tier" in result
        assert result["churn_probability"] == pytest.approx(0.7, abs=0.01)
        assert result["churn_label"] == 1  # 0.7 >= threshold 0.45
        assert result["risk_tier"] == "HIGH"

    def test_risk_tier_assignment(self):
        from src.api.model_loader import assign_risk_tier
        assert assign_risk_tier(0.10) == "LOW"
        assert assign_risk_tier(0.30) == "MEDIUM"
        assert assign_risk_tier(0.60) == "HIGH"
        assert assign_risk_tier(0.80) == "CRITICAL"

    def test_revenue_at_risk_calculation(self):
        from src.api.model_loader import ModelLoader
        import numpy as np
        import pandas as pd

        loader = ModelLoader()
        artifact = make_mock_model()
        loader.model = artifact["model"]
        loader.feature_cols = artifact["feature_cols"]
        loader.threshold = artifact["threshold"]

        mock_X = pd.DataFrame(np.zeros((1, 5)), columns=artifact["feature_cols"])
        payload = {**SAMPLE_PAYLOAD, "monthly_charges": 100.0}

        with patch.object(loader, "_build_features", return_value=mock_X):
            result = loader.predict_single(payload)

        # revenue_at_risk = monthly_charges * churn_probability
        expected = round(100.0 * 0.7, 2)
        assert result["monthly_revenue_at_risk"] == pytest.approx(expected, abs=0.5)


class TestDataValidation:
    """Test the data validation module."""

    def test_valid_data_passes(self):
        import numpy as np
        from src.data_engineering.data_validation import DataValidator

        df = __import__("tests.test_features", fromlist=["make_sample_df"]).make_sample_df(500)
        validator = DataValidator(df, "test")
        validator.expect_row_count_between(100)
        validator.expect_column_between("churned", 0, 1)
        assert validator.report() is True

    def test_duplicate_ids_fail(self):
        import pandas as pd
        from src.data_engineering.data_validation import DataValidator

        df = pd.DataFrame({"subscriber_id": ["A", "A", "B"], "churned": [0, 1, 0]})
        validator = DataValidator(df, "test")
        validator.expect_unique_column("subscriber_id")
        assert validator.report() is False

    def test_out_of_range_values_fail(self):
        import pandas as pd
        from src.data_engineering.data_validation import DataValidator

        df = pd.DataFrame({"churned": [0, 1, 5, -1]})  # 5 and -1 are invalid
        validator = DataValidator(df, "test")
        validator.expect_column_between("churned", 0, 1)
        assert validator.report() is False
