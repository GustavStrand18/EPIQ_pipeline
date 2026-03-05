"""
test_features.py
----------------
Unit tests for feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
    build_features,
)


def make_price_df(n_hours: int = 500) -> pd.DataFrame:
    """Helper: create a realistic hourly price DataFrame."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")
    prices = np.random.uniform(20, 100, size=n_hours)
    return pd.DataFrame({"price_eur_mwh": prices}, index=idx)


# --- Time features ---

def test_time_features_columns_exist():
    df = make_price_df(100)
    result = add_time_features(df)
    for col in ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos"]:
        assert col in result.columns


def test_hour_range():
    df = make_price_df(100)
    result = add_time_features(df)
    assert result["hour"].between(0, 23).all()


def test_cyclical_encoding_bounds():
    df = make_price_df(100)
    result = add_time_features(df)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


# --- Lag features ---

def test_lag_features_columns_exist():
    df = make_price_df(200)
    result = add_lag_features(df)
    for col in ["price_lag_24h", "price_lag_48h", "price_lag_168h"]:
        assert col in result.columns


def test_lag_24h_correct_value():
    df = make_price_df(200)
    result = add_lag_features(df)
    # Row at index 24 should have lag_24h equal to the price at index 0
    assert result["price_lag_24h"].iloc[24] == df["price_eur_mwh"].iloc[0]


def test_first_rows_have_nan_lags():
    df = make_price_df(200)
    result = add_lag_features(df)
    assert result["price_lag_24h"].iloc[:24].isna().all()


# --- Rolling features ---

def test_rolling_features_columns_exist():
    df = make_price_df(200)
    result = add_rolling_features(df)
    for col in ["rolling_mean_24h", "rolling_std_24h", "rolling_min_24h", "rolling_max_24h"]:
        assert col in result.columns


# --- Full pipeline ---

def test_build_features_drops_nan():
    df = make_price_df(500)
    result = build_features(df, drop_na=True)
    assert result.isna().sum().sum() == 0


def test_build_features_shape():
    df = make_price_df(500)
    result = build_features(df, drop_na=True)
    # Should have more columns than the original
    assert result.shape[1] > df.shape[1]