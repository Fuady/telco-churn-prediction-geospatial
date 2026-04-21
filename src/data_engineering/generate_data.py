"""
src/data_engineering/generate_data.py
──────────────────────────────────────
Generates a realistic synthetic telecom subscriber dataset with:
  - Subscriber demographics & contract attributes
  - Monthly usage metrics (calls, data, SMS)
  - Network quality KPIs per subscriber (RSRP, RSRQ, throughput)
  - Geographic coordinates (lat/lon within configurable bounding box)
  - Churn label with realistic correlations

Usage:
    python src/data_engineering/generate_data.py
    python src/data_engineering/generate_data.py --n_subscribers 100000 --output data/raw/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import yaml

# ── Make src importable when run directly ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_subscribers(
    n: int,
    geo_bounds: dict,
    churn_rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate the core subscriber table with all features."""

    logger.info(f"Generating {n:,} subscriber records...")

    # ── Demographics ─────────────────────────────────────────────────────────
    subscriber_ids = [f"SUB_{i:07d}" for i in range(1, n + 1)]

    contract_types = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n,
        p=[0.50, 0.28, 0.22],
    )
    payment_methods = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    internet_service = rng.choice(
        ["fiber_optic", "DSL", "none"],
        size=n,
        p=[0.44, 0.34, 0.22],
    )
    phone_service = rng.choice([1, 0], size=n, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == 1, rng.choice([1, 0], size=n, p=[0.50, 0.50]), 0
    )
    senior_citizen = rng.choice([1, 0], size=n, p=[0.16, 0.84])

    # ── Tenure & Charges ─────────────────────────────────────────────────────
    # Month-to-month customers have shorter tenure (they churn more)
    tenure_map = {
        "month-to-month": {"mu": 18, "sigma": 12, "clip": (1, 72)},
        "one-year": {"mu": 36, "sigma": 18, "clip": (12, 72)},
        "two-year": {"mu": 54, "sigma": 18, "clip": (24, 72)},
    }
    tenure_months = np.array([
        np.clip(
            rng.normal(tenure_map[c]["mu"], tenure_map[c]["sigma"]),
            *tenure_map[c]["clip"],
        ).astype(int)
        for c in contract_types
    ])

    base_charge = rng.uniform(20, 45, size=n)
    internet_addon = np.where(internet_service == "fiber_optic", rng.uniform(30, 50, n),
                     np.where(internet_service == "DSL", rng.uniform(15, 30, n), 0))
    phone_addon = phone_service * rng.uniform(5, 20, n)
    monthly_charges = (base_charge + internet_addon + phone_addon).round(2)
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.95, 1.05, n)).round(2)

    # ── Usage Metrics ─────────────────────────────────────────────────────────
    data_usage_gb = np.clip(rng.lognormal(mean=2.0, sigma=0.8, size=n), 0.1, 100).round(2)
    call_minutes_monthly = np.clip(rng.normal(350, 150, n), 0, 1200).round(0)
    sms_monthly = np.clip(rng.normal(80, 60, n), 0, 500).round(0)
    international_calls = rng.choice([1, 0], size=n, p=[0.28, 0.72])
    tech_support_calls = np.clip(rng.poisson(1.2, n), 0, 10)

    # ── Network Quality KPIs ──────────────────────────────────────────────────
    # RSRP: Reference Signal Received Power (dBm). Good: > -80, Poor: < -110
    rsrp_avg = np.clip(rng.normal(-95, 15, n), -130, -60).round(1)
    # RSRQ: Reference Signal Received Quality (dB). Good: > -10, Poor: < -15
    rsrq_avg = np.clip(rng.normal(-11, 3, n), -20, -3).round(1)
    # Throughput (Mbps)
    dl_throughput_mbps = np.clip(rng.lognormal(2.5, 0.8, n), 0.5, 150).round(2)
    # Dropped call rate (%)
    call_drop_rate = np.clip(rng.beta(1.5, 10, n) * 20, 0, 20).round(2)
    # Call drops per month
    call_drops_monthly = np.clip(
        (call_drop_rate / 100 * call_minutes_monthly / 3).astype(int), 0, 20
    )
    # Network outage minutes per month
    outage_minutes_monthly = np.clip(rng.poisson(30, n), 0, 300)

    # ── Geography ─────────────────────────────────────────────────────────────
    # Cluster subscribers into realistic urban clusters (Jakarta metropolitan)
    n_clusters = 20
    cluster_centers_lat = rng.uniform(
        geo_bounds["lat_min"], geo_bounds["lat_max"], n_clusters
    )
    cluster_centers_lon = rng.uniform(
        geo_bounds["lon_min"], geo_bounds["lon_max"], n_clusters
    )
    # Weight clusters by urban density (Poisson-driven)
    cluster_weights = rng.dirichlet(np.ones(n_clusters) * 2)
    cluster_assignment = rng.choice(n_clusters, size=n, p=cluster_weights)

    latitudes = np.array([
        np.clip(
            rng.normal(cluster_centers_lat[c], 0.05),
            geo_bounds["lat_min"],
            geo_bounds["lat_max"],
        )
        for c in cluster_assignment
    ]).round(6)
    longitudes = np.array([
        np.clip(
            rng.normal(cluster_centers_lon[c], 0.05),
            geo_bounds["lon_min"],
            geo_bounds["lon_max"],
        )
        for c in cluster_assignment
    ]).round(6)

    # ── Churn Label (with realistic feature correlations) ─────────────────────
    # Churn probability is a function of real business drivers
    churn_logit = (
        -2.5                                          # base intercept
        + 1.5 * (contract_types == "month-to-month").astype(float)
        + 0.8 * (contract_types == "one-year").astype(float)
        - 0.02 * tenure_months                        # longer tenure = lower churn
        + 0.015 * monthly_charges                     # higher charges = more likely to churn
        - 0.5 * (internet_service == "fiber_optic").astype(float)  # better service = less churn
        + 0.04 * call_drops_monthly                   # drops drive churn
        + 0.03 * outage_minutes_monthly / 10          # outages drive churn
        + 0.03 * (rsrq_avg + 11) * -1                 # worse signal = more churn
        + 0.4 * (payment_methods == "electronic_check").astype(float)
        + 0.3 * senior_citizen
        + 0.05 * tech_support_calls
        + rng.normal(0, 0.3, n)                       # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    # Scale to target churn rate
    churn_prob = churn_prob / churn_prob.mean() * churn_rate
    churn_prob = np.clip(churn_prob, 0, 1)
    churned = rng.binomial(1, churn_prob).astype(int)

    logger.info(f"Actual churn rate: {churned.mean():.2%}")

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "subscriber_id": subscriber_ids,
        "snapshot_date": pd.Timestamp("2024-01-01"),
        # Demographics
        "senior_citizen": senior_citizen,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "international_calls": international_calls,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        # Tenure & charges
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        # Usage
        "data_usage_gb": data_usage_gb,
        "call_minutes_monthly": call_minutes_monthly,
        "sms_monthly": sms_monthly,
        "tech_support_calls": tech_support_calls,
        # Network quality
        "rsrp_avg": rsrp_avg,
        "rsrq_avg": rsrq_avg,
        "dl_throughput_mbps": dl_throughput_mbps,
        "call_drop_rate_pct": call_drop_rate,
        "call_drops_monthly": call_drops_monthly,
        "outage_minutes_monthly": outage_minutes_monthly,
        # Geography
        "latitude": latitudes,
        "longitude": longitudes,
        # Target
        "churned": churned,
        "churn_probability_true": churn_prob.round(4),  # for diagnostics only
    })

    return df


def generate_network_towers(
    n_towers: int,
    geo_bounds: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic cell tower locations (approximates OpenCelliD structure)."""
    logger.info(f"Generating {n_towers} synthetic cell towers...")

    tower_ids = [f"TOWER_{i:05d}" for i in range(1, n_towers + 1)]
    radio_types = rng.choice(["LTE", "NR", "UMTS"], size=n_towers, p=[0.55, 0.25, 0.20])

    return pd.DataFrame({
        "tower_id": tower_ids,
        "radio": radio_types,
        "mcc": 510,         # Indonesia MCC
        "mnc": rng.choice([10, 8, 1], size=n_towers),  # Telkomsel, XL, Indosat
        "latitude": rng.uniform(geo_bounds["lat_min"], geo_bounds["lat_max"], n_towers).round(6),
        "longitude": rng.uniform(geo_bounds["lon_min"], geo_bounds["lon_max"], n_towers).round(6),
        "range_m": rng.choice([500, 1000, 2000, 5000], size=n_towers, p=[0.3, 0.4, 0.2, 0.1]),
        "samples": rng.randint(50, 5000, size=n_towers),
        "created": pd.Timestamp("2023-01-01"),
        "updated": pd.Timestamp("2024-01-01"),
    })


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic telecom dataset")
    parser.add_argument("--n_subscribers", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed or config["data_generation"]["random_seed"]
    rng = np.random.default_rng(seed)

    geo_bounds = config["data_generation"]["geo_bounds"]
    churn_rate = config["data_generation"]["churn_rate"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate subscribers ──────────────────────────────────────────────────
    df_subscribers = generate_subscribers(
        n=args.n_subscribers,
        geo_bounds=geo_bounds,
        churn_rate=churn_rate,
        rng=rng,
    )

    sub_path = output_dir / "subscribers.parquet"
    df_subscribers.to_parquet(sub_path, index=False)
    logger.success(f"Saved {len(df_subscribers):,} subscribers → {sub_path}")

    # Also save a CSV sample for easy inspection
    sample_path = output_dir / "subscribers_sample.csv"
    df_subscribers.head(1000).to_csv(sample_path, index=False)
    logger.info(f"Saved 1,000-row CSV sample → {sample_path}")

    # ── Generate towers ───────────────────────────────────────────────────────
    n_towers = max(200, args.n_subscribers // 200)
    df_towers = generate_network_towers(n_towers, geo_bounds, rng)

    tower_path = output_dir / "cell_towers.parquet"
    df_towers.to_parquet(tower_path, index=False)
    logger.success(f"Saved {len(df_towers):,} cell towers → {tower_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Subscribers  : {len(df_subscribers):,}")
    print(f"  Churned      : {df_subscribers['churned'].sum():,} ({df_subscribers['churned'].mean():.1%})")
    print(f"  Cell Towers  : {len(df_towers):,}")
    print(f"  Output dir   : {output_dir.resolve()}")
    print("=" * 60)
    print("\nColumn summary:")
    print(df_subscribers.describe().to_string())


if __name__ == "__main__":
    main()
