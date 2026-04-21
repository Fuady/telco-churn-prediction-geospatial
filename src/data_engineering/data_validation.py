"""
src/data_engineering/data_validation.py
────────────────────────────────────────
Data quality checks for the subscriber dataset.
Implements lightweight validation rules (Great Expectations inspired).
Catches issues early in the pipeline before they corrupt model training.

Usage:
    python src/data_engineering/data_validation.py --input data/raw/subscribers.parquet
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ValidationResult:
    rule: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR or WARNING

    def __str__(self):
        icon = "✓" if self.passed else ("✗" if self.severity == "ERROR" else "⚠")
        return f"  [{icon}] {self.rule}: {self.message}"


class DataValidator:
    """Run a suite of validation rules on a DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self.results: list[ValidationResult] = []

    def _add(self, rule: str, passed: bool, msg: str, severity: str = "ERROR"):
        self.results.append(ValidationResult(rule, passed, msg, severity))

    # ── Schema rules ──────────────────────────────────────────────────────────
    def expect_columns(self, expected_cols: list[str]):
        missing = [c for c in expected_cols if c not in self.df.columns]
        self._add(
            "expect_columns",
            len(missing) == 0,
            f"Missing columns: {missing}" if missing else f"All {len(expected_cols)} required columns present",
        )
        return self

    def expect_no_nulls(self, columns: list[str]):
        for col in columns:
            if col not in self.df.columns:
                continue
            null_count = self.df[col].isnull().sum()
            self._add(
                f"no_nulls:{col}",
                null_count == 0,
                f"{null_count:,} nulls found ({null_count/len(self.df):.2%})" if null_count else "No nulls",
            )
        return self

    # ── Range rules ───────────────────────────────────────────────────────────
    def expect_column_between(self, col: str, min_val: float, max_val: float):
        if col not in self.df.columns:
            return self
        out_of_range = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
        self._add(
            f"range:{col}",
            out_of_range == 0,
            f"{out_of_range:,} values outside [{min_val}, {max_val}]" if out_of_range else f"All values in [{min_val}, {max_val}]",
            severity="WARNING" if out_of_range < len(self.df) * 0.01 else "ERROR",
        )
        return self

    def expect_column_positive(self, col: str):
        if col not in self.df.columns:
            return self
        negatives = (self.df[col] < 0).sum()
        self._add(
            f"positive:{col}",
            negatives == 0,
            f"{negatives:,} negative values" if negatives else "All positive",
        )
        return self

    # ── Categorical rules ──────────────────────────────────────────────────────
    def expect_column_values_in_set(self, col: str, valid_values: set):
        if col not in self.df.columns:
            return self
        unexpected = ~self.df[col].isin(valid_values)
        n = unexpected.sum()
        self._add(
            f"valid_values:{col}",
            n == 0,
            f"{n:,} unexpected values: {self.df.loc[unexpected, col].unique()[:5]}" if n else f"All values in {valid_values}",
        )
        return self

    # ── Distribution rules ──────────────────────────────────────────────────
    def expect_churn_rate_between(self, target_col: str, min_rate: float, max_rate: float):
        if target_col not in self.df.columns:
            return self
        rate = self.df[target_col].mean()
        self._add(
            "churn_rate",
            min_rate <= rate <= max_rate,
            f"Churn rate = {rate:.2%} (expected {min_rate:.0%}–{max_rate:.0%})",
            severity="WARNING",
        )
        return self

    def expect_row_count_between(self, min_rows: int, max_rows: int = None):
        n = len(self.df)
        ok = n >= min_rows and (max_rows is None or n <= max_rows)
        self._add(
            "row_count",
            ok,
            f"{n:,} rows (expected >= {min_rows:,})",
        )
        return self

    def expect_unique_column(self, col: str):
        if col not in self.df.columns:
            return self
        dups = self.df[col].duplicated().sum()
        self._add(
            f"unique:{col}",
            dups == 0,
            f"{dups:,} duplicates found" if dups else "All values unique",
        )
        return self

    # ── Report ───────────────────────────────────────────────────────────────
    def report(self) -> bool:
        """Print full validation report. Returns True if all ERROR rules passed."""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {self.name}")
        print(f"{'='*60}")

        errors = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]
        passed = [r for r in self.results if r.passed]

        for r in self.results:
            print(r)

        print(f"\n  Summary: {len(passed)} passed, {len(warnings)} warnings, {len(errors)} errors")
        print("=" * 60)

        if errors:
            logger.error(f"Validation FAILED: {len(errors)} critical errors")
            return False
        elif warnings:
            logger.warning(f"Validation passed with {len(warnings)} warnings")
            return True
        else:
            logger.success("Validation PASSED: all rules satisfied")
            return True


def validate_subscribers(df: pd.DataFrame) -> bool:
    """Full validation suite for the subscriber dataset."""

    REQUIRED_COLS = [
        "subscriber_id", "tenure_months", "monthly_charges", "total_charges",
        "data_usage_gb", "call_minutes_monthly", "sms_monthly",
        "rsrp_avg", "rsrq_avg", "dl_throughput_mbps",
        "call_drop_rate_pct", "call_drops_monthly", "outage_minutes_monthly",
        "latitude", "longitude", "contract_type", "payment_method",
        "internet_service", "churned",
    ]

    validator = DataValidator(df, name="subscribers.parquet")

    return (
        validator
        .expect_row_count_between(min_rows=1000)
        .expect_columns(REQUIRED_COLS)
        .expect_unique_column("subscriber_id")
        .expect_no_nulls(["subscriber_id", "churned", "latitude", "longitude",
                          "monthly_charges", "tenure_months"])
        .expect_column_positive("tenure_months")
        .expect_column_positive("monthly_charges")
        .expect_column_positive("total_charges")
        .expect_column_between("churned", 0, 1)
        .expect_column_between("latitude", -11.0, 6.0)      # Indonesia
        .expect_column_between("longitude", 95.0, 141.0)
        .expect_column_between("rsrp_avg", -140, -40)
        .expect_column_between("rsrq_avg", -25, 0)
        .expect_column_between("call_drop_rate_pct", 0, 100)
        .expect_column_values_in_set(
            "contract_type",
            {"month-to-month", "one-year", "two-year"}
        )
        .expect_column_values_in_set(
            "internet_service",
            {"fiber_optic", "DSL", "none"}
        )
        .expect_churn_rate_between("churned", 0.05, 0.35)
        .report()
    )


def main():
    parser = argparse.ArgumentParser(description="Validate subscriber dataset")
    parser.add_argument("--input", type=str, default="data/raw/subscribers.parquet")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        logger.error(f"File not found: {path}")
        logger.info("Run data generation first: python src/data_engineering/generate_data.py")
        sys.exit(1)

    logger.info(f"Loading {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    ok = validate_subscribers(df)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
