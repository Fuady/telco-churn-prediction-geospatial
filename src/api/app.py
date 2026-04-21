"""
src/api/app.py
───────────────
FastAPI REST API for real-time churn prediction.

Endpoints:
  POST /predict          — Score a single subscriber
  POST /predict/batch    — Score a batch of subscribers (up to 1000)
  GET  /geo-risk-map     — Return GeoJSON risk grid
  GET  /health           — Health check
  GET  /model-info       — Model metadata

Usage:
    uvicorn src.api.app:app --reload --port 8000
    # Swagger UI: http://localhost:8000/docs
"""

import sys
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import (
    SubscriberInput,
    ChurnPrediction,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.model_loader import ModelLoader

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Real-time churn prediction with geospatial risk scoring. "
        "Combines subscriber attributes, usage patterns, and network quality KPIs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loader (singleton) ──────────────────────────────────────────────────
loader = ModelLoader()


@app.on_event("startup")
async def startup():
    logger.info("Loading model on startup...")
    loader.load()
    logger.info("Model loaded successfully.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check — returns model status."""
    return HealthResponse(
        status="healthy" if loader.is_loaded() else "degraded",
        model_loaded=loader.is_loaded(),
        model_version=loader.model_version,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns model metadata and feature list."""
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=loader.model_type,
        model_version=loader.model_version,
        n_features=len(loader.feature_cols),
        threshold=loader.threshold,
        feature_names=loader.feature_cols[:20],  # preview first 20
    )


@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
async def predict(subscriber: SubscriberInput):
    """
    Score a single subscriber for churn risk.

    Returns churn probability, risk tier, H3 cell, and top contributing factors.
    """
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = loader.predict_single(subscriber.dict())
        return ChurnPrediction(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Score a batch of subscribers (max 1000 per request).
    """
    if not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.subscribers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000")

    try:
        results = loader.predict_batch([s.dict() for s in request.subscribers])
        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            high_risk_count=sum(1 for r in results if r.get("risk_tier") in ["HIGH", "CRITICAL"]),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/geo-risk-map", tags=["Geospatial"])
async def geo_risk_map():
    """
    Returns the current H3 hexagonal churn risk grid as GeoJSON.
    Use this in Kepler.gl or any GeoJSON-compatible map tool.
    """
    geojson_path = Path("data/processed/risk_grid.geojson")
    if not geojson_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Risk map not found. Run: python src/models/geo_risk_map.py"
        )
    with open(geojson_path) as f:
        return json.load(f)


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Telecom Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
