"""
src/data_engineering/ingest_opencellid.py
──────────────────────────────────────────
Downloads and processes REAL cell tower data from OpenCelliD (opencellid.org).
OpenCelliD is the world's largest open database of cell towers.

HOW TO GET THE DATA (free):
  1. Go to https://opencellid.org/downloads.php
  2. Register for a free account
  3. Get your API token from your profile page
  4. Set OPENCELLID_TOKEN in your .env file
  5. Run: python src/data_engineering/ingest_opencellid.py --country ID

Alternative (no signup):
  - Download the full database as CSV from the website manually
  - Place it at data/external/cell_towers_raw.csv
  - Run: python src/data_engineering/ingest_opencellid.py --local data/external/cell_towers_raw.csv

Data schema:
  radio, mcc, net, area, cell, unit, lon, lat, range, samples, changeable, created, updated, averageSignal

Usage:
    python src/data_engineering/ingest_opencellid.py --country ID --token YOUR_TOKEN
"""

import os
import sys
import argparse
import requests
import gzip
import shutil
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# MCC codes for country filtering
COUNTRY_MCC = {
    "ID": [510],   # Indonesia
    "SG": [525],   # Singapore
    "MY": [502],   # Malaysia
    "AU": [505],   # Australia
    "US": [310, 311, 312],  # USA
    "GB": [234, 235],       # UK
}

# Indonesia bounding box
DEFAULT_BOUNDS = {
    "lat_min": -11.0,
    "lat_max": 6.0,
    "lon_min": 95.0,
    "lon_max": 141.0,
}


def download_opencellid(token: str, output_dir: Path) -> Path:
    """
    Download full OpenCelliD database.
    File is ~1GB compressed; filtered result is much smaller.
    """
    url = f"https://download.opencellid.org/ocid/downloads?token={token}&type=full&file=cell_towers.csv.gz"
    gz_path = output_dir / "cell_towers.csv.gz"

    logger.info(f"Downloading OpenCelliD database (~1GB)...")
    logger.info(f"URL: {url}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}% ({downloaded / 1e6:.0f} MB)", end="")
    print()
    logger.success(f"Downloaded to {gz_path}")
    return gz_path


def decompress_and_filter(
    gz_path: Path,
    output_dir: Path,
    country: str,
    bounds: dict,
) -> pd.DataFrame:
    """Decompress and filter towers by country MCC and bounding box."""

    mcc_list = COUNTRY_MCC.get(country, [])
    logger.info(f"Filtering towers for country={country}, MCC={mcc_list}")

    csv_path = output_dir / "cell_towers_raw.csv"
    logger.info("Decompressing...")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    logger.info("Reading and filtering CSV (this may take a minute for large files)...")
    cols = ["radio", "mcc", "net", "area", "cell", "unit",
            "lon", "lat", "range", "samples", "changeable",
            "created", "updated", "averageSignal"]

    chunks = []
    for chunk in pd.read_csv(csv_path, names=cols, chunksize=500_000, low_memory=False):
        # Filter by MCC (country)
        if mcc_list:
            chunk = chunk[chunk["mcc"].isin(mcc_list)]
        # Filter by bounding box
        chunk = chunk[
            (chunk["lat"] >= bounds["lat_min"]) &
            (chunk["lat"] <= bounds["lat_max"]) &
            (chunk["lon"] >= bounds["lon_min"]) &
            (chunk["lon"] <= bounds["lon_max"])
        ]
        if len(chunk) > 0:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    logger.info(f"Filtered: {len(df):,} towers for {country}")
    return df


def process_towers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Clean and enrich tower data, convert to GeoDataFrame."""
    df = df.copy()

    # Rename to consistent schema
    df = df.rename(columns={
        "net": "mnc",
        "area": "lac",
        "cell": "cell_id",
        "lon": "longitude",
        "lat": "latitude",
        "range": "range_m",
        "averageSignal": "avg_signal",
    })

    # Drop invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        (df["latitude"].between(-90, 90)) &
        (df["longitude"].between(-180, 180))
    ]

    # Convert timestamps
    for col in ["created", "updated"]:
        df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")

    # Add tower type label
    radio_map = {"GSM": "2G", "UMTS": "3G", "LTE": "4G", "NR": "5G"}
    df["generation"] = df["radio"].map(radio_map).fillna("Unknown")

    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf


def load_from_local(csv_path: str) -> pd.DataFrame:
    """Load from a locally downloaded CSV (no API token needed)."""
    logger.info(f"Loading local file: {csv_path}")
    cols = ["radio", "mcc", "net", "area", "cell", "unit",
            "lon", "lat", "range", "samples", "changeable",
            "created", "updated", "averageSignal"]
    df = pd.read_csv(csv_path, names=cols, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenCelliD tower data")
    parser.add_argument("--country", type=str, default="ID", help="ISO country code")
    parser.add_argument("--token", type=str, default=None,
                        help="OpenCelliD API token (or set OPENCELLID_TOKEN env var)")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to locally downloaded cell_towers.csv (skip download)")
    parser.add_argument("--output", type=str, default="data/external")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("OPENCELLID_TOKEN")

    bounds = DEFAULT_BOUNDS

    if args.local:
        # Use local file
        df_raw = load_from_local(args.local)
        mcc_list = COUNTRY_MCC.get(args.country, [])
        if mcc_list:
            df_raw = df_raw[df_raw["mcc"].isin(mcc_list)]
        df_raw = df_raw[
            (df_raw["lat"].between(bounds["lat_min"], bounds["lat_max"])) &
            (df_raw["lon"].between(bounds["lon_min"], bounds["lon_max"]))
        ]
    elif token:
        gz_path = download_opencellid(token, output_dir)
        df_raw = decompress_and_filter(gz_path, output_dir, args.country, bounds)
    else:
        logger.error(
            "No data source provided. Either:\n"
            "  1. Set --token YOUR_OPENCELLID_TOKEN\n"
            "  2. Set OPENCELLID_TOKEN in your .env file\n"
            "  3. Use --local path/to/cell_towers.csv\n"
            "\nGet a free token at: https://opencellid.org/downloads.php"
        )
        sys.exit(1)

    # Process
    gdf = process_towers(df_raw)

    # Save outputs
    parquet_path = output_dir / f"cell_towers_{args.country}.parquet"
    geojson_path = output_dir / f"cell_towers_{args.country}.geojson"

    gdf.drop(columns=["geometry"]).to_parquet(parquet_path, index=False)
    gdf.to_file(geojson_path, driver="GeoJSON")

    logger.success(f"Saved Parquet → {parquet_path}")
    logger.success(f"Saved GeoJSON → {geojson_path}")

    print("\n" + "=" * 50)
    print(f"OpenCelliD Towers — {args.country}")
    print("=" * 50)
    print(gdf["radio"].value_counts().to_string())
    print(f"\nTotal towers: {len(gdf):,}")


if __name__ == "__main__":
    main()
