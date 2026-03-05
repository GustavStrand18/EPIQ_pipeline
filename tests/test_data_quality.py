"""
test_data_quality.py
--------------------
Unit tests for data quality checks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_quality import (
    check_missing_values,
    check_duplicate_timestamps,
    check_time_gaps,
    check_outliers,
    run_quality_checks,
)


def make_clean_df(n_hours: int = 48) -> pd.DataFrame:
    """Helper: create a clean hourly price DataFrame."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="Europe/Copenhagen")
    prices = np.random.uniform(20, 100, size=n_hours)
    return pd.DataFrame({"price_eur_mwh": prices}, index=idx)


# --- Missing values ---

def test_no_missing_values():
    df = make_clean_df()
    assert check_missing_values(df) == 0


def test_detects_missing_values():
    df = make_clean_df()
    df.iloc[5, 0] = np.nan
    df.iloc[10, 0] = np.nan
    assert check_missing_values(df) == 2


# --- Duplicate timestamps ---

def test_no_duplicates():
    df = make_clean_df()
    assert check_duplicate_timestamps(df) == 0


def test_detects_duplicates():
    df = make_clean_df()
    df = pd.concat([df, df.iloc[[0]]])  # Add a duplicate row
    assert check_duplicate_timestamps(df) == 1


# --- Time gaps ---

def test_no_time_gaps():
    df = make_clean_df(48)
    gaps = check_time_gaps(df)
    assert len(gaps) == 0


def test_detects_time_gap():
    df = make_clean_df(48)
    # Drop 3 consecutive hours to create a gap
    df = df.drop(df.index[10:13])
    gaps = check_time_gaps(df)
    assert len(gaps) == 1


# --- Outliers ---

def test_no_outliers_in_normal_data():
    df = make_clean_df(200)
    outliers = check_outliers(df, z_threshold=4.0)
    assert len(outliers) == 0


def test_detects_extreme_price_spike():
    df = make_clean_df(200)
    df.iloc[50, 0] = 99999  # Extreme price spike
    outliers = check_outliers(df, z_threshold=4.0)
    assert len(outliers) >= 1


# --- Full report ---

def test_clean_data_passes_quality_check():
    df = make_clean_df(168)  # One week of data
    report = run_quality_checks(df, zone="TEST")
    assert report.passed is True


def test_dirty_data_fails_quality_check():
    df = make_clean_df(168)
    df.iloc[5, 0] = np.nan  # Introduce a missing value
    report = run_quality_checks(df, zone="TEST")
    assert report.passed is False
