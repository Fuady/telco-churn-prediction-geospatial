"""
src/features/feature_pipeline.py
──────────────────────────────────
Orchestrates the full feature engineering pipeline:
  1. Load raw subscriber data
  2. Add H3 geospatial indexes
  3. Add tower proximity features
  4. Add OSM POI features (if available)
  5. Engineer subscriber features
  6. Engineer network features
  7. Add H3 aggregate features
  8. Encode categoricals
  9. Train/test split
  10. Save processed datasets

Usage:
    python src/features/feature_pipeline.py
    python src/features/feature_pipeline.py --input data/raw/ --output data/processed/
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.geospatial_features import (
    add_h3_indexes,
    add_tower_features,
    add_poi_features,
    add_h3_aggregate_features,
)
from src.features.subscriber_features import (
    SubscriberFeatureEngineer,
    NetworkFeatureEngineer,
    encode_categoricals,
)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading raw subscriber data...")
    df = pd.read_parquet(input_dir / "subscribers.parquet")
    logger.info(f"  {len(df):,} subscribers loaded")

    logger.info("Loading cell tower data...")
    towers_path = input_dir / "cell_towers.parquet"
    towers = pd.read_parquet(towers_path)
    logger.info(f"  {len(towers):,} towers loaded")

    # Load POI data if available
    pois_path = input_dir.parent / "external" / "osm_pois_jakarta.parquet"
    pois = None
    if pois_path.exists():
        pois = pd.read_parquet(pois_path)
        logger.info(f"  {len(pois):,} POIs loaded")
    else:
        logger.warning("POI data not found — skipping POI features. "
                       "Run: python src/data_engineering/ingest_osm.py")

    # ── 2. Geospatial features ────────────────────────────────────────────────
    logger.info("Step 2: Adding H3 indexes...")
    df = add_h3_indexes(df, resolutions=[7, 8])

    logger.info("Step 3: Adding tower features...")
    df = add_tower_features(df, towers)

    if pois is not None:
        logger.info("Step 4: Adding POI features...")
        df = add_poi_features(df, pois)

    # ── 3. Subscriber & Network features ─────────────────────────────────────
    logger.info("Step 5: Engineering subscriber features...")
    sub_eng = SubscriberFeatureEngineer()
    df = sub_eng.fit_transform(df)

    logger.info("Step 6: Engineering network features...")
    net_eng = NetworkFeatureEngineer()
    df = net_eng.fit_transform(df)

    # ── 4. H3 aggregate features (AFTER individual features) ─────────────────
    logger.info("Step 7: Adding H3 aggregate features...")
    # IMPORTANT: compute aggregates on full dataset, but use leave-one-out
    # during training to avoid target leakage from cell_churn_rate
    df = add_h3_aggregate_features(df, h3_col="h3_r8")

    # ── 5. Encode categoricals ────────────────────────────────────────────────
    logger.info("Step 8: Encoding categorical features...")
    df = encode_categoricals(df)

    # ── 6. Train/test split ───────────────────────────────────────────────────
    logger.info("Step 9: Splitting train/test...")
    cfg = config["features"]
    target = cfg["target_column"]
    split_ratio = cfg["train_test_split"]
    seed = config["data_generation"]["random_seed"]

    df_train, df_test = train_test_split(
        df, test_size=1 - split_ratio, stratify=df[target], random_state=seed
    )

    logger.info(f"  Train: {len(df_train):,} rows ({df_train[target].mean():.2%} churn)")
    logger.info(f"  Test:  {len(df_test):,} rows ({df_test[target].mean():.2%} churn)")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    logger.info("Step 10: Saving processed data...")
    train_path = output_dir / "features_train.parquet"
    test_path = output_dir / "features_test.parquet"
    full_path = output_dir / "features_full.parquet"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    df.to_parquet(full_path, index=False)

    # Save feature list (used by model training)
    feature_cols = [c for c in df.columns if c not in [
        target, "subscriber_id", "snapshot_date",
        "churn_probability_true",     # diagnostic only
        "h3_r7", "h3_r8",             # string IDs, not model features
        "nearest_tower_radio",        # encoded separately
    ]]
    feature_list = output_dir / "feature_columns.txt"
    feature_list.write_text("\n".join(feature_cols))

    logger.success(f"Saved train → {train_path}")
    logger.success(f"Saved test  → {test_path}")
    logger.success(f"Saved full  → {full_path}")
    logger.success(f"Features: {len(feature_cols)} columns")

    print("\n" + "=" * 60)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total features : {len(feature_cols)}")
    print(f"  Train rows     : {len(df_train):,}")
    print(f"  Test rows      : {len(df_test):,}")
    print(f"  Train churn    : {df_train[target].mean():.2%}")
    print(f"  Test churn     : {df_test[target].mean():.2%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(
        config=config,
        input_dir=Path(args.input),
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
