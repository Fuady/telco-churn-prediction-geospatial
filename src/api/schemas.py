"""
src/api/schemas.py
───────────────────
Pydantic v2 request/response models for the churn prediction API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class SubscriberInput(BaseModel):
    """Input payload for a single subscriber churn prediction."""

    subscriber_id: str = Field(..., example="SUB_0000001")

    # Subscriber attributes
    senior_citizen: int = Field(0, ge=0, le=1, example=0)
    phone_service: int = Field(1, ge=0, le=1, example=1)
    multiple_lines: int = Field(0, ge=0, le=1, example=0)
    internet_service: str = Field("fiber_optic", example="fiber_optic",
                                   description="fiber_optic | DSL | none")
    international_calls: int = Field(0, ge=0, le=1, example=0)
    contract_type: str = Field("month-to-month", example="month-to-month",
                                description="month-to-month | one-year | two-year")
    payment_method: str = Field("electronic_check", example="electronic_check")
    tenure_months: int = Field(..., ge=0, le=120, example=18)
    monthly_charges: float = Field(..., ge=0, example=65.0)
    total_charges: float = Field(..., ge=0, example=1170.0)

    # Usage
    data_usage_gb: float = Field(..., ge=0, example=12.5)
    call_minutes_monthly: float = Field(300.0, ge=0, example=300.0)
    sms_monthly: float = Field(50.0, ge=0, example=50.0)
    tech_support_calls: int = Field(0, ge=0, example=1)

    # Network quality
    rsrp_avg: float = Field(-95.0, description="dBm. Good: >-80, Poor: <-110", example=-95.0)
    rsrq_avg: float = Field(-11.0, description="dB. Good: >-10, Poor: <-15", example=-11.0)
    dl_throughput_mbps: float = Field(20.0, ge=0, example=20.0)
    call_drop_rate_pct: float = Field(2.0, ge=0, le=100, example=2.0)
    call_drops_monthly: int = Field(1, ge=0, example=1)
    outage_minutes_monthly: float = Field(30.0, ge=0, example=30.0)

    # Geography
    latitude: float = Field(..., ge=-90, le=90, example=-6.2088)
    longitude: float = Field(..., ge=-180, le=180, example=106.8456)

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        valid = {"month-to-month", "one-year", "two-year"}
        if v not in valid:
            raise ValueError(f"contract_type must be one of {valid}")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet(cls, v):
        valid = {"fiber_optic", "DSL", "none"}
        if v not in valid:
            raise ValueError(f"internet_service must be one of {valid}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "subscriber_id": "SUB_0000001",
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
        }


class ChurnPrediction(BaseModel):
    """Response model for a single churn prediction."""
    subscriber_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_label: int = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    h3_cell: Optional[str] = None
    top_factors: List[str] = Field(default_factory=list,
                                    description="Top 3 features driving churn risk")
    monthly_revenue_at_risk: float = Field(0.0, description="Monthly revenue if subscriber churns")


class BatchPredictionRequest(BaseModel):
    subscribers: List[SubscriberInput] = Field(..., max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: Optional[str]
    n_features: int
    threshold: float
    feature_names: List[str]
