"""
src/models/geo_risk_map.py
───────────────────────────
Aggregates subscriber-level churn predictions onto an H3 hex grid,
creating a geospatial risk map for marketing and network operations teams.

Output:
  - GeoJSON with H3 polygons coloured by churn risk tier
  - Parquet with per-H3-cell risk metrics
  - Interactive Folium HTML map

Usage:
    python src/models/geo_risk_map.py
    python src/models/geo_risk_map.py --output data/processed/risk_grid.geojson
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import folium
import json
import yaml
from loguru import logger

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def h3_to_polygon(h3_index: str):
    """Convert H3 index to Shapely polygon."""
    if not H3_AVAILABLE:
        return None
    try:
        from shapely.geometry import Polygon
        boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        return Polygon(boundary)
    except Exception:
        return None


def run_batch_prediction(
    df: pd.DataFrame,
    model_artifact: dict,
) -> pd.DataFrame:
    """Score all subscribers using the trained model."""
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    threshold = model_artifact["threshold"]

    # Align feature columns
    X = df[feature_cols].fillna(-999)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    df = df.copy()
    df["churn_probability"] = prob.round(4)
    df["churn_predicted"] = pred

    return df


def assign_risk_tier(prob: float, tiers: dict) -> str:
    """Map a probability to a risk tier label."""
    for tier, (lo, hi) in tiers.items():
        if lo <= prob < hi:
            return tier
    return "CRITICAL"


def aggregate_to_h3(
    df: pd.DataFrame,
    h3_col: str,
    min_subscribers: int,
) -> pd.DataFrame:
    """Aggregate subscriber predictions to H3 cell level."""
    agg = df.groupby(h3_col).agg(
        subscriber_count=(h3_col, "count"),
        predicted_churners=("churn_predicted", "sum"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_rsrq=("rsrq_avg", "mean"),
        avg_throughput=("dl_throughput_mbps", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_outage_minutes=("outage_minutes_monthly", "mean"),
        pct_month_to_month=("contract_risk_score", lambda x: (x == 3).mean()),
    ).reset_index()

    # Filter cells with too few subscribers (unreliable estimates)
    agg = agg[agg["subscriber_count"] >= min_subscribers].copy()

    agg["predicted_churn_rate"] = (
        agg["predicted_churners"] / agg["subscriber_count"]
    ).round(4)

    agg["avg_churn_probability"] = agg["avg_churn_probability"].round(4)
    agg["avg_rsrq"] = agg["avg_rsrq"].round(2)
    agg["avg_throughput"] = agg["avg_throughput"].round(2)
    agg["avg_monthly_charges"] = agg["avg_monthly_charges"].round(2)
    agg["avg_outage_minutes"] = agg["avg_outage_minutes"].round(1)
    agg["pct_month_to_month"] = agg["pct_month_to_month"].round(3)

    # Estimated revenue at risk (avg ARPU × predicted churners)
    arpu = agg["avg_monthly_charges"]
    agg["estimated_revenue_at_risk"] = (arpu * agg["predicted_churners"]).round(2)

    logger.info(f"H3 risk grid: {len(agg):,} cells ({h3_col})")
    return agg


def build_geodataframe(risk_df: pd.DataFrame, h3_col: str, tiers: dict) -> gpd.GeoDataFrame:
    """Convert H3 risk dataframe to GeoDataFrame with polygon geometries."""
    if not H3_AVAILABLE:
        logger.warning("H3 library not available — skipping geometry creation")
        return gpd.GeoDataFrame(risk_df)

    logger.info("Building H3 polygon geometries...")
    geometries = [h3_to_polygon(h) for h in risk_df[h3_col]]
    valid = [g is not None for g in geometries]

    risk_df = risk_df[valid].copy()
    geometries = [g for g in geometries if g is not None]

    gdf = gpd.GeoDataFrame(risk_df, geometry=geometries, crs="EPSG:4326")

    # Assign risk tiers
    gdf["risk_tier"] = gdf["avg_churn_probability"].apply(
        lambda p: assign_risk_tier(p, tiers)
    )

    # Colour mapping for viz
    tier_colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#8e44ad",
    }
    gdf["color"] = gdf["risk_tier"].map(tier_colors)

    return gdf


def create_folium_map(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Create interactive Folium choropleth map of churn risk."""
    # Centre on the data
    centre_lat = gdf.geometry.centroid.y.mean()
    centre_lon = gdf.geometry.centroid.x.mean()

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    # Choropleth layer
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        try:
            geojson = gpd.GeoDataFrame([row], geometry="geometry").to_json()
            folium.GeoJson(
                geojson,
                style_function=lambda feature, color=row["color"]: {
                    "fillColor": color,
                    "color": "#333",
                    "weight": 0.5,
                    "fillOpacity": 0.6,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["subscriber_count", "predicted_churners",
                            "avg_churn_probability", "risk_tier",
                            "avg_rsrq", "estimated_revenue_at_risk"],
                    aliases=["Subscribers", "Predicted Churners",
                             "Avg Churn Prob", "Risk Tier",
                             "Avg RSRQ (dB)", "Revenue at Risk ($)"],
                    localize=True,
                ),
            ).add_to(m)
        except Exception:
            pass

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:#fff; padding:12px; border-radius:8px;
                border:1px solid #ccc; font-size:13px;">
      <b>Churn Risk</b><br>
      <span style="background:#2ecc71;padding:2px 10px;margin:2px;">&nbsp;</span> Low (< 25%)<br>
      <span style="background:#f39c12;padding:2px 10px;margin:2px;">&nbsp;</span> Medium (25–50%)<br>
      <span style="background:#e74c3c;padding:2px 10px;margin:2px;">&nbsp;</span> High (50–70%)<br>
      <span style="background:#8e44ad;padding:2px 10px;margin:2px;">&nbsp;</span> Critical (> 70%)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))
    logger.success(f"Interactive map saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate geospatial churn risk map")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_path", type=str, default="data/models/churn_model_xgboost.pkl")
    parser.add_argument("--input", type=str, default="data/processed/features_full.parquet")
    parser.add_argument("--output", type=str, default="data/processed/risk_grid.geojson")
    args = parser.parse_args()

    config = load_config(args.config)
    geo_cfg = config["geo_risk"]

    # ── Load model and data ───────────────────────────────────────────────────
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run train.py first.")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
    model_artifact = joblib.load(model_path)

    data_path = Path(args.input)
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}. Run feature_pipeline.py first.")
        sys.exit(1)

    logger.info(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    # ── Run predictions ───────────────────────────────────────────────────────
    logger.info("Running batch predictions...")
    df_scored = run_batch_prediction(df, model_artifact)

    overall_churn_rate = df_scored["churn_predicted"].mean()
    logger.info(f"Overall predicted churn rate: {overall_churn_rate:.2%}")

    # ── Aggregate to H3 grid ──────────────────────────────────────────────────
    h3_col = f"h3_r{geo_cfg['risk_map_resolution']}"
    if h3_col not in df_scored.columns:
        logger.warning(f"{h3_col} not found, falling back to h3_r8")
        h3_col = "h3_r8"

    risk_df = aggregate_to_h3(
        df_scored, h3_col, geo_cfg["min_subscribers_per_cell"]
    )

    # ── Build GeoDataFrame ────────────────────────────────────────────────────
    tiers = {k: tuple(v) for k, v in geo_cfg["risk_tiers"].items()}
    gdf = build_geodataframe(risk_df, h3_col, tiers)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GeoJSON
    if len(gdf) > 0 and gdf.geometry is not None:
        gdf.to_file(output_path, driver="GeoJSON")
        logger.success(f"Risk grid GeoJSON → {output_path}")

    # Parquet (no geometry column)
    parquet_path = output_path.with_suffix(".parquet")
    risk_df.to_parquet(parquet_path, index=False)
    logger.success(f"Risk grid Parquet → {parquet_path}")

    # Interactive HTML map
    map_path = output_path.with_name("churn_risk_map.html")
    if H3_AVAILABLE and len(gdf) > 0:
        create_folium_map(gdf, map_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    tier_counts = risk_df.copy()
    tier_counts["risk_tier"] = risk_df["avg_churn_probability"].apply(
        lambda p: assign_risk_tier(p, tiers)
    )

    print("\n" + "=" * 60)
    print("GEO RISK MAP COMPLETE")
    print("=" * 60)
    print(f"  H3 resolution  : {geo_cfg['risk_map_resolution']}")
    print(f"  Total H3 cells : {len(risk_df):,}")
    print(f"  Total subscribers scored: {len(df_scored):,}")
    print(f"\n  Risk tier breakdown:")
    for tier, count in tier_counts["risk_tier"].value_counts().items():
        pct = count / len(tier_counts) * 100
        print(f"    {tier:10s}: {count:5d} cells ({pct:.1f}%)")
    rev_at_risk = risk_df["estimated_revenue_at_risk"].sum()
    print(f"\n  Total revenue at risk: ${rev_at_risk:,.0f}/month")
    print("=" * 60)


if __name__ == "__main__":
    main()
