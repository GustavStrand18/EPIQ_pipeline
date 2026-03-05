"""
features.py
-----------
Feature engineering for electricity price forecasting.
Transforms raw hourly price data into model-ready features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based time features.
    Electricity prices follow strong hourly, daily, and seasonal patterns.
    """
    df = df.copy()
    local = df.index.tz_convert("Europe/Copenhagen")

    df["hour"]          = local.hour
    df["day_of_week"]   = local.dayofweek      # 0=Monday, 6=Sunday
    df["month"]         = local.month
    df["is_weekend"]    = (local.dayofweek >= 5).astype(int)
    df["is_monday"]     = (local.dayofweek == 0).astype(int)

    # Cyclical encoding — avoids the jump between hour 23 and hour 0
    df["hour_sin"]      = np.sin(2 * np.pi * local.hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * local.hour / 24)
    df["month_sin"]     = np.sin(2 * np.pi * local.month / 12)
    df["month_cos"]     = np.cos(2 * np.pi * local.month / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged price features.
    In day-ahead markets, the most useful lags are 24h and 48h
    because prices are set one day in advance.
    """
    df = df.copy()
    price = df["price_eur_mwh"]

    df["price_lag_24h"]  = price.shift(24)   # Same hour yesterday
    df["price_lag_48h"]  = price.shift(48)   # Same hour 2 days ago
    df["price_lag_168h"] = price.shift(168)  # Same hour last week

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics.
    Captures recent price trends and volatility.
    """
    df = df.copy()
    price = df["price_eur_mwh"]

    # Rolling means — use shift(1) to avoid data leakage
    df["rolling_mean_24h"]  = price.shift(1).rolling(24).mean()
    df["rolling_mean_168h"] = price.shift(1).rolling(168).mean()

    # Rolling std — captures price volatility
    df["rolling_std_24h"]   = price.shift(1).rolling(24).std()

    # Daily min/max — useful signal for within-day spread
    df["rolling_min_24h"]   = price.shift(1).rolling(24).min()
    df["rolling_max_24h"]   = price.shift(1).rolling(24).max()

    return df


def add_daily_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical average price by hour of day.
    Captures the typical daily price shape (peak/off-peak).
    """
    df = df.copy()
    local = df.index.tz_convert("Europe/Copenhagen")

    # Average price per hour of day across the whole dataset
    hourly_avg = (
        df.groupby(local.hour)["price_eur_mwh"]
        .transform("mean")
    )
    df["hist_avg_by_hour"] = hourly_avg

    return df


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Run all feature engineering steps and return a model-ready DataFrame.

    Args:
        df:       Raw price DataFrame with 'price_eur_mwh' column
        drop_na:  Whether to drop rows with NaN (from lags/rolling windows)

    Returns:
        Feature-enriched DataFrame
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_daily_profile_features(df)

    if drop_na:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        print(f"  Dropped {dropped} rows with NaN (from lag/rolling warmup)")

    print(f"  Feature matrix shape: {df.shape}")
    print(f"  Features: {[c for c in df.columns if c != 'price_eur_mwh']}")
    return df


def save_processed(df: pd.DataFrame, zone: str) -> Path:
    """Save processed feature DataFrame to data/processed/"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = PROCESSED_DATA_DIR / f"features_{zone}.csv"
    df.to_csv(filename)
    print(f"  Saved to {filename}")
    return filename


def load_processed(zone: str = "DK1") -> pd.DataFrame:
    """Load processed feature data from CSV."""
    filename = PROCESSED_DATA_DIR / f"features_{zone}.csv"
    if not filename.exists():
        raise FileNotFoundError(
            f"No processed data found for {zone}. Run build_features() first."
        )
    df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
    return df


if __name__ == "__main__":
    from data_ingestion import load_raw

    for zone in ["DK1", "DK2"]:
        print(f"\nBuilding features for {zone}...")
        df = load_raw(zone=zone)
        df_features = build_features(df)
        save_processed(df_features, zone=zone)