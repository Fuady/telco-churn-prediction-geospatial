"""
src/features/geospatial_features.py
─────────────────────────────────────
Geospatial feature engineering using H3 hexagonal indexing.

Features created:
  - H3 cell index (resolution 8 and 7) for each subscriber
  - Distance to nearest cell tower (km)
  - Number of towers within 1km / 2km radius
  - H3-cell-level aggregate stats (avg churn, network quality, density)
  - Distance to nearest OSM POI per category
  - POI density within H3 cell
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    logger.warning("h3 not installed — install with: pip install h3")


def lat_lon_to_h3(lat: float, lon: float, resolution: int) -> str:
    """Convert lat/lon to H3 cell index."""
    if H3_AVAILABLE:
        return h3.geo_to_h3(lat, lon, resolution)
    # Fallback: simple grid approximation if h3 not installed
    grid_lat = round(lat * (2 ** resolution), 0) / (2 ** resolution)
    grid_lon = round(lon * (2 ** resolution), 0) / (2 ** resolution)
    return f"fake_h3_{grid_lat}_{grid_lon}"


def add_h3_indexes(df: pd.DataFrame, resolutions: list[int] = [7, 8]) -> pd.DataFrame:
    """Add H3 cell indexes at multiple resolutions."""
    logger.info(f"Adding H3 indexes at resolutions: {resolutions}")
    df = df.copy()
    for res in resolutions:
        col = f"h3_r{res}"
        df[col] = df.apply(
            lambda row: lat_lon_to_h3(row["latitude"], row["longitude"], res), axis=1
        )
        n_cells = df[col].nunique()
        logger.info(f"  Resolution {res}: {n_cells:,} unique H3 cells")
    return df


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    """Vectorised haversine distance (km) from each point to a single target."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_tower_features(
    df: pd.DataFrame,
    towers_df: pd.DataFrame,
    radius_km: float = 2.0,
) -> pd.DataFrame:
    """
    For each subscriber, compute:
      - dist_to_nearest_tower_km
      - towers_within_1km
      - towers_within_2km
      - nearest_tower_radio (2G/3G/4G/5G)
    """
    logger.info(f"Computing tower distance features for {len(df):,} subscribers...")

    sub_lats = df["latitude"].values
    sub_lons = df["longitude"].values
    tower_lats = towers_df["latitude"].values
    tower_lons = towers_df["longitude"].values

    dist_to_nearest = np.full(len(df), np.inf)
    nearest_radio = np.full(len(df), "", dtype=object)
    towers_1km = np.zeros(len(df), dtype=int)
    towers_2km = np.zeros(len(df), dtype=int)

    # Process in chunks to avoid memory issues with large datasets
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        end = min(i + chunk_size, len(df))
        chunk_lats = sub_lats[i:end]
        chunk_lons = sub_lons[i:end]

        # Distance matrix: (chunk_size x n_towers)
        dlat = np.radians(tower_lats[None, :] - chunk_lats[:, None])
        dlon = np.radians(tower_lons[None, :] - chunk_lons[:, None])
        lat1_r = np.radians(chunk_lats[:, None])
        lat2_r = np.radians(tower_lats[None, :])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        dists = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # (chunk x towers)

        nearest_idx = np.argmin(dists, axis=1)
        dist_to_nearest[i:end] = dists[np.arange(end - i), nearest_idx]

        radio_arr = towers_df["radio"].values if "radio" in towers_df.columns else towers_df.get("generation", pd.Series(["LTE"] * len(towers_df))).values
        nearest_radio[i:end] = radio_arr[nearest_idx]

        towers_1km[i:end] = (dists <= 1.0).sum(axis=1)
        towers_2km[i:end] = (dists <= 2.0).sum(axis=1)

        if (i // chunk_size) % 5 == 0:
            logger.info(f"  Processed {end:,}/{len(df):,} subscribers")

    df = df.copy()
    df["dist_to_nearest_tower_km"] = dist_to_nearest.round(3)
    df["nearest_tower_radio"] = nearest_radio
    df["towers_within_1km"] = towers_1km
    df["towers_within_2km"] = towers_2km

    logger.info("Tower features complete.")
    return df


def add_poi_features(
    df: pd.DataFrame,
    pois_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each subscriber, compute:
      - dist_to_nearest_{poi_type}_km (for each POI category)
      - poi_density_h3r8 (POI count in same H3 cell)
    """
    if pois_df is None or len(pois_df) == 0:
        logger.warning("No POI data available, skipping POI features")
        return df

    logger.info("Computing POI distance features...")
    df = df.copy()

    poi_types = pois_df["poi_type"].unique()
    sub_lats = df["latitude"].values
    sub_lons = df["longitude"].values

    for poi_type in poi_types:
        subset = pois_df[pois_df["poi_type"] == poi_type]
        if len(subset) == 0:
            continue

        poi_lats = subset["latitude"].values
        poi_lons = subset["longitude"].values

        min_dists = []
        chunk_size = 5000
        for i in range(0, len(df), chunk_size):
            end = min(i + chunk_size, len(df))
            dlat = np.radians(poi_lats[None, :] - sub_lats[i:end, None])
            dlon = np.radians(poi_lons[None, :] - sub_lons[i:end, None])
            lat1_r = np.radians(sub_lats[i:end, None])
            lat2_r = np.radians(poi_lats[None, :])
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
            dists = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            min_dists.extend(np.min(dists, axis=1).tolist())

        df[f"dist_to_{poi_type}_km"] = np.array(min_dists).round(3)
        logger.info(f"  {poi_type}: done")

    return df


def add_h3_aggregate_features(df: pd.DataFrame, h3_col: str = "h3_r8") -> pd.DataFrame:
    """
    Compute within-H3-cell aggregate statistics.
    These capture neighbourhood-level effects beyond individual subscriber features.
    """
    logger.info(f"Computing H3 aggregate features at {h3_col}...")
    df = df.copy()

    agg = df.groupby(h3_col).agg(
        cell_subscriber_count=(h3_col, "count"),
        cell_avg_rsrq=("rsrq_avg", "mean"),
        cell_avg_rsrp=("rsrp_avg", "mean"),
        cell_avg_throughput=("dl_throughput_mbps", "mean"),
        cell_avg_outage=("outage_minutes_monthly", "mean"),
        cell_avg_monthly_charges=("monthly_charges", "mean"),
        cell_avg_data_usage=("data_usage_gb", "mean"),
        cell_churn_rate=("churned", "mean"),       # NOTE: use with care — leakage risk in train set
    ).reset_index()

    df = df.merge(agg, on=h3_col, how="left")

    logger.info(f"  H3 aggregate features added: {agg.shape[1] - 1} new columns")
    return df


if __name__ == "__main__":
    # Quick smoke test
    import yaml
    config = yaml.safe_load(open("configs/config.yaml"))

    df = pd.read_parquet("data/raw/subscribers.parquet")
    towers = pd.read_parquet("data/raw/cell_towers.parquet")

    logger.info(f"Loaded {len(df):,} subscribers, {len(towers):,} towers")

    df = add_h3_indexes(df, resolutions=[7, 8])
    df = add_tower_features(df, towers)
    df = add_h3_aggregate_features(df, h3_col="h3_r8")

    print(df[["subscriber_id", "h3_r8", "dist_to_nearest_tower_km",
              "towers_within_1km", "cell_churn_rate"]].head(10))
