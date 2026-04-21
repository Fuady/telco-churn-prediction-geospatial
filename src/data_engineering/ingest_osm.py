"""
src/data_engineering/ingest_osm.py
────────────────────────────────────
Downloads Points of Interest (POIs) from OpenStreetMap via the Overpass API.
Completely FREE — no account or API key needed.

POI categories collected:
  - Commercial: shops, malls, supermarkets
  - Transport: bus stops, train stations, airports
  - Education: schools, universities
  - Healthcare: hospitals, clinics
  - Recreation: parks, stadiums

These are used as geospatial enrichment features for the churn model:
  - Distance to nearest commercial area (high data usage correlation)
  - Distance to transport hubs (commuter pattern proxy)
  - POI density within H3 cell

Usage:
    python src/data_engineering/ingest_osm.py --city "Jakarta"
    python src/data_engineering/ingest_osm.py --bbox -6.4,-7.0,107.2,106.4
"""

import argparse
import time
import json
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from loguru import logger


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Named city bounding boxes [south, west, north, east]
CITY_BOUNDS = {
    "Jakarta":   [-6.4,  106.65, -5.95, 107.10],
    "Surabaya":  [-7.40, 112.55, -7.10, 112.85],
    "Bandung":   [-6.98, 107.50, -6.83, 107.72],
    "Singapore": [1.20,  103.60,  1.48, 104.05],
    "Kuala Lumpur": [2.95, 101.55, 3.28, 101.85],
}

POI_QUERIES = {
    "shopping": 'node["shop"~"mall|supermarket|convenience|department_store"]',
    "transport": 'node["public_transport"~"station|stop_position"]["railway"~"station|subway_entrance"]',
    "education": 'node["amenity"~"school|university|college"]',
    "healthcare": 'node["amenity"~"hospital|clinic|pharmacy"]',
    "recreation": 'node["leisure"~"park|stadium|sports_centre"]',
    "food_beverage": 'node["amenity"~"restaurant|cafe|fast_food"]',
}


def build_overpass_query(bbox: list, poi_type: str, query_filter: str) -> str:
    """Build an Overpass QL query for a given bounding box and POI type."""
    south, west, north, east = bbox
    return f"""
    [out:json][timeout:60];
    (
      {query_filter}({south},{west},{north},{east});
    );
    out body;
    """


def fetch_pois(bbox: list, poi_type: str, query_filter: str) -> list:
    """Fetch POIs from Overpass API for one category."""
    query = build_overpass_query(bbox, poi_type, query_filter)
    logger.info(f"Fetching {poi_type} POIs from Overpass API...")

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=120,
            headers={"User-Agent": "TelecomChurnResearch/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        elements = data.get("elements", [])
        logger.info(f"  {poi_type}: {len(elements):,} POIs found")
        return elements
    except requests.exceptions.Timeout:
        logger.warning(f"  {poi_type}: request timed out — skipping")
        return []
    except Exception as e:
        logger.warning(f"  {poi_type}: error ({e}) — skipping")
        return []


def elements_to_geodataframe(elements: list, poi_type: str) -> gpd.GeoDataFrame:
    """Convert Overpass JSON elements to a GeoDataFrame."""
    rows = []
    for el in elements:
        if el.get("type") == "node" and "lat" in el and "lon" in el:
            tags = el.get("tags", {})
            rows.append({
                "osm_id": el["id"],
                "poi_type": poi_type,
                "name": tags.get("name", ""),
                "latitude": el["lat"],
                "longitude": el["lon"],
            })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["osm_id", "poi_type", "name", "latitude", "longitude", "geometry"]
        )

    df = pd.DataFrame(rows)
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(description="Download OSM POIs for geospatial enrichment")
    parser.add_argument("--city", type=str, default="Jakarta",
                        help=f"City name. Available: {list(CITY_BOUNDS.keys())}")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Custom bbox: south,west,north,east (e.g. -6.4,-7.0,107.2,106.4)")
    parser.add_argument("--output", type=str, default="data/external")
    parser.add_argument("--poi_types", nargs="+",
                        default=list(POI_QUERIES.keys()),
                        help="POI categories to download")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve bounding box
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
    elif args.city in CITY_BOUNDS:
        bbox = CITY_BOUNDS[args.city]
    else:
        logger.error(f"Unknown city '{args.city}'. Use --bbox or choose from: {list(CITY_BOUNDS.keys())}")
        return

    logger.info(f"Area: {args.city} | bbox: {bbox}")

    all_gdfs = []
    for poi_type in args.poi_types:
        if poi_type not in POI_QUERIES:
            logger.warning(f"Unknown POI type: {poi_type}, skipping")
            continue

        gdf = elements_to_geodataframe(
            fetch_pois(bbox, poi_type, POI_QUERIES[poi_type]),
            poi_type,
        )
        if len(gdf) > 0:
            all_gdfs.append(gdf)

        # Be polite to the free API — don't hammer it
        time.sleep(2)

    if not all_gdfs:
        logger.error("No POI data collected.")
        return

    gdf_all = pd.concat(all_gdfs, ignore_index=True)
    gdf_all = gpd.GeoDataFrame(gdf_all, crs="EPSG:4326")

    city_slug = args.city.lower().replace(" ", "_")
    out_parquet = output_dir / f"osm_pois_{city_slug}.parquet"
    out_geojson = output_dir / f"osm_pois_{city_slug}.geojson"

    gdf_all.drop(columns=["geometry"]).to_parquet(out_parquet, index=False)
    gdf_all.to_file(out_geojson, driver="GeoJSON")

    logger.success(f"Saved {len(gdf_all):,} POIs → {out_parquet}")
    logger.success(f"Saved GeoJSON → {out_geojson}")

    print("\nPOI type breakdown:")
    print(gdf_all["poi_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
