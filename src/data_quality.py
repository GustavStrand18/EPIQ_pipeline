"""
data_quality.py
---------------
Data quality checks for electricity price data.
Checks for missing values, outliers, and time gaps.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class QualityReport:
    """Holds results of all data quality checks."""
    zone: str
    total_records: int
    missing_values: int
    duplicate_timestamps: int
    time_gaps: List[str] = field(default_factory=list)
    outliers: List[str] = field(default_factory=list)
    passed: bool = True

    def print_summary(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"\n--- Data Quality Report: {self.zone} {status} ---")
        print(f"  Total records:        {self.total_records}")
        print(f"  Missing values:       {self.missing_values}")
        print(f"  Duplicate timestamps: {self.duplicate_timestamps}")
        print(f"  Time gaps found:      {len(self.time_gaps)}")
        if self.time_gaps:
            for gap in self.time_gaps[:5]:  # Show first 5 gaps
                print(f"    - {gap}")
        print(f"  Outliers found:       {len(self.outliers)}")
        if self.outliers:
            for o in self.outliers[:5]:
                print(f"    - {o}")


def check_missing_values(df: pd.DataFrame) -> int:
    """Count missing price values."""
    return int(df["price_eur_mwh"].isna().sum())


def check_duplicate_timestamps(df: pd.DataFrame) -> int:
    """Count duplicate timestamps in the index."""
    return int(df.index.duplicated().sum())


def check_time_gaps(df: pd.DataFrame, expected_freq: str = "h") -> List[str]:
    """
    Find gaps in the time series where hours are missing entirely.
    Returns a list of human-readable gap descriptions.
    """
    gaps = []
    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=expected_freq,
        tz=df.index.tz,
    )
    missing_timestamps = expected_index.difference(df.index)

    if len(missing_timestamps) == 0:
        return gaps

    # Group consecutive missing timestamps into ranges
    missing_series = pd.Series(missing_timestamps)
    gap_start = missing_series.iloc[0]
    prev = missing_series.iloc[0]

    for ts in missing_series.iloc[1:]:
        if (ts - prev).total_seconds() > 3600:  # More than 1 hour apart = new gap
            gaps.append(f"{gap_start} → {prev}")
            gap_start = ts
        prev = ts
    gaps.append(f"{gap_start} → {prev}")

    return gaps


def check_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> List[str]:
    """
    Flag prices more than z_threshold standard deviations from the mean.
    Electricity markets do have extreme spikes, so we use a high threshold.
    """
    prices = df["price_eur_mwh"]
    mean = prices.mean()
    std = prices.std()
    z_scores = (prices - mean) / std
    outlier_mask = z_scores.abs() > z_threshold

    return [
        f"{ts}: {prices[ts]:.2f} EUR/MWh (z={z_scores[ts]:.1f})"
        for ts in df.index[outlier_mask]
        if pd.notna(prices[ts])
    ]


def run_quality_checks(df: pd.DataFrame, zone: str = "DK1") -> QualityReport:
    """Run all quality checks and return a QualityReport."""
    missing = check_missing_values(df)
    duplicates = check_duplicate_timestamps(df)
    gaps = check_time_gaps(df)
    outliers = check_outliers(df)

    passed = missing == 0 and duplicates == 0 and len(gaps) == 0

    report = QualityReport(
        zone=zone,
        total_records=len(df),
        missing_values=missing,
        duplicate_timestamps=duplicates,
        time_gaps=gaps,
        outliers=outliers,
        passed=passed,
    )
    report.print_summary()
    return report


if __name__ == "__main__":
    from data_ingestion import load_raw

    for zone in ["DK1", "DK2"]:
        df = load_raw(zone=zone)
        run_quality_checks(df, zone=zone)