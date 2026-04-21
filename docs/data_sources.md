# Data Sources Guide

This document explains every data source used in this project, how to obtain it, and what it contributes to the model.

---

## 1. Synthetic Subscriber Dataset (Auto-generated)

**No download needed.** Run the generator:

```bash
python src/data_engineering/generate_data.py --n_subscribers 50000
```

### What it produces (`data/raw/subscribers.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `subscriber_id` | str | Unique subscriber ID |
| `tenure_months` | int | Months as a customer |
| `monthly_charges` | float | Monthly bill ($) |
| `contract_type` | str | month-to-month / one-year / two-year |
| `internet_service` | str | fiber_optic / DSL / none |
| `rsrp_avg` | float | Avg signal power (dBm). Good: > -80 |
| `rsrq_avg` | float | Avg signal quality (dB). Good: > -10 |
| `dl_throughput_mbps` | float | Average download speed |
| `call_drop_rate_pct` | float | % of calls that drop |
| `outage_minutes_monthly` | float | Monthly network outage duration |
| `latitude`, `longitude` | float | Subscriber location (Jakarta metro area) |
| `churned` | int | Target: 1 = churned, 0 = retained |

The generator uses realistic correlations (e.g., bad RSRQ → higher churn, month-to-month → 3× more churn than annual contract) based on published telecom research.

---

## 2. OpenCelliD — Real Cell Tower Locations

**Website:** https://opencellid.org  
**Cost:** Free  
**Signup:** Required (free account)  
**License:** Creative Commons 4.0 Attribution

### How to get it

#### Option A — API download (recommended)
1. Register at https://opencellid.org/register
2. Get your API token from your profile page
3. Add to `.env`: `OPENCELLID_TOKEN=your_token_here`
4. Run:
```bash
python src/data_engineering/ingest_opencellid.py --country ID
```

#### Option B — Manual download
1. Go to https://opencellid.org/downloads.php
2. Download `cell_towers.csv.gz` (≈1GB)
3. Extract and place at `data/external/cell_towers_raw.csv`
4. Run:
```bash
python src/data_engineering/ingest_opencellid.py --local data/external/cell_towers_raw.csv --country ID
```

### Schema
```
radio, mcc, net (mnc), area (lac), cell (cell_id), lon, lat, range_m, samples, created, updated
```

### What it adds to the model
- `dist_to_nearest_tower_km` — proxy for network coverage quality
- `towers_within_1km` / `towers_within_2km` — network density
- `nearest_tower_radio` — 2G/3G/4G/5G coverage type

---

## 3. OpenStreetMap POIs — Points of Interest

**Website:** https://overpass-api.de  
**Cost:** Free — no account needed  
**License:** ODbL (Open Database License)

### How to get it

```bash
# Download POIs for Jakarta
python src/data_engineering/ingest_osm.py --city "Jakarta"

# Or custom bounding box
python src/data_engineering/ingest_osm.py --bbox -6.4,106.65,-5.95,107.10
```

**Available cities:** Jakarta, Surabaya, Bandung, Singapore, Kuala Lumpur

### POI categories collected
| Category | OSM filter | Model contribution |
|----------|-----------|-------------------|
| Shopping | shop=mall/supermarket | High-traffic zones (data usage) |
| Transport | railway=station | Commuter pattern proxy |
| Education | amenity=school/university | Demographic proxy |
| Healthcare | amenity=hospital/clinic | Mobility pattern |
| Recreation | leisure=park/stadium | Footfall proxy |

### What it adds to the model
- `dist_to_shopping_km` — proximity to commercial zones
- `dist_to_transport_km` — commuter area indicator
- `dist_to_education_km` — demographic signal

---

## 4. WorldPop Population Density (Optional)

**Website:** https://www.worldpop.org  
**Cost:** Free  
**Format:** GeoTIFF raster, 100m resolution

### How to get it
1. Go to https://hub.worldpop.org/geodata/listing?id=69
2. Select Indonesia (IDN), year 2020
3. Download the 100m population count GeoTIFF
4. Place at `data/external/idn_ppp_2020_1km_Aggregated.tif`

This adds population density as a raster-based feature that can be spatially joined to H3 cells.

---

## 5. GADM Administrative Boundaries (Optional)

**Website:** https://gadm.org/download_country.html  
**Cost:** Free  
**Format:** GeoJSON / Shapefile

Used to annotate H3 cells with province/district names for reporting.

```bash
# Download Indonesia shapefile
wget "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IDN_2.json.zip" -O data/external/gadm_idn.zip
unzip data/external/gadm_idn.zip -d data/external/
```

---

## Data Dictionary Summary

| Dataset | Rows | Columns | Used for |
|---------|------|---------|----------|
| subscribers.parquet | 50,000 | 25 | Training / inference target |
| cell_towers_{country}.parquet | ~50,000 | 12 | Tower proximity features |
| osm_pois_{city}.parquet | ~20,000 | 6 | POI distance features |
| risk_grid.parquet | ~5,000 H3 cells | 10 | Geo risk map output |
| risk_grid.geojson | ~5,000 H3 cells | 10 | Dashboard / Kepler.gl |

---

## Privacy & Compliance Note

The synthetic subscriber dataset contains no real personal data. All lat/lon coordinates are randomly generated within the Jakarta metro bounding box. In a real production deployment, subscriber location data would be:

- Anonymised at the H3 cell level (not exact addresses)
- Subject to local privacy regulations (OJK in Indonesia, GDPR in EU)
- Aggregated to groups of ≥10 subscribers before any public output
