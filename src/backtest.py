"""
backtest.py
-----------
Backtesting engine for a simple electricity trading strategy.

IMPORTANT — avoiding look-ahead bias:
We use a strict train/test split. The model is trained on the first 75% of
data and backtested only on the remaining 25% it has never seen.
This gives honest, realistic performance estimates.

Strategy logic:
- Each day we have 24 hourly price forecasts for the next day
- We buy (go long) in the N cheapest forecast hours
- We sell (go short) in the N most expensive forecast hours
- PnL is calculated using actual prices, not forecast prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from typing import Tuple

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_monday",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "rolling_mean_24h", "rolling_mean_168h", "rolling_std_24h",
    "rolling_min_24h", "rolling_max_24h", "hist_avg_by_hour",
]


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.75):
    """
    Split data chronologically — never randomly.
    Training on future data to predict the past is look-ahead bias.
    """
    split = int(len(df) * train_ratio)
    train = df.iloc[:split]
    test  = df.iloc[split:]
    print(f"  Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} rows)")
    print(f"  Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} rows)")
    return train, test


def generate_signals(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    n_hours: int = 6,
) -> pd.DataFrame:
    """
    Generate buy/sell signals for each hour based on forecasted prices.

    For each day:
    - The N hours with the lowest forecasted price  → BUY signal  (+1)
    - The N hours with the highest forecasted price → SELL signal (-1)
    - All other hours → no position (0)
    """
    df = df.copy()
    df["forecast"] = model.predict(df[FEATURE_COLS])
    df["signal"]   = 0

    local_index = df.index.tz_convert("Europe/Copenhagen")
    df["date"] = local_index.date

    for date, day_df in df.groupby("date"):
        if len(day_df) < 24:
            continue

        sorted_by_forecast = day_df["forecast"].sort_values()
        buy_hours  = sorted_by_forecast.index[:n_hours]
        sell_hours = sorted_by_forecast.index[-n_hours:]

        df.loc[buy_hours,  "signal"] = 1
        df.loc[sell_hours, "signal"] = -1

    df = df.drop(columns=["date"])
    return df


def calculate_pnl(df: pd.DataFrame, mwh_per_trade: float = 1.0) -> pd.DataFrame:
    """
    Calculate hourly and cumulative PnL from trading signals.

    PnL per hour = signal × (actual_price - daily_mean_price) × MWh
    This represents the spread captured vs simply trading at the daily average.
    """
    df = df.copy()

    local_index = df.index.tz_convert("Europe/Copenhagen")
    df["date"] = local_index.date
    daily_mean = df.groupby("date")["price_eur_mwh"].transform("mean")

    df["pnl_eur"] = df["signal"] * (df["price_eur_mwh"] - daily_mean) * mwh_per_trade * -1
    df["cumulative_pnl_eur"] = df["pnl_eur"].cumsum()

    df = df.drop(columns=["date"])
    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key trading performance metrics."""
    trades = df[df["signal"] != 0].copy()
    winning_trades = trades[trades["pnl_eur"] > 0]

    local_index = df.index.tz_convert("Europe/Copenhagen")
    df_copy = df.copy()
    df_copy["date"] = local_index.date
    daily_pnl = df_copy.groupby("date")["pnl_eur"].sum()

    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
              if daily_pnl.std() > 0 else 0)

    cumulative = df["cumulative_pnl_eur"]
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_drawdown = drawdown.min()

    return {
        "total_pnl_eur":     round(df["pnl_eur"].sum(), 2),
        "n_trades":          len(trades),
        "win_rate_pct":      round(len(winning_trades) / len(trades) * 100, 1),
        "sharpe_ratio":      round(sharpe, 2),
        "max_drawdown_eur":  round(max_drawdown, 2),
        "avg_daily_pnl_eur": round(daily_pnl.mean(), 2),
    }


def plot_backtest(df: pd.DataFrame, metrics: dict, zone: str = "DK1"):
    """Plot cumulative PnL and trading signals, save to models/"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(df.index, df["cumulative_pnl_eur"], color="steelblue", linewidth=1.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.fill_between(df.index, df["cumulative_pnl_eur"], 0,
                     where=df["cumulative_pnl_eur"] >= 0, alpha=0.2, color="green")
    ax1.fill_between(df.index, df["cumulative_pnl_eur"], 0,
                     where=df["cumulative_pnl_eur"] < 0,  alpha=0.2, color="red")
    ax1.set_title(
        f"{zone} Backtest (out-of-sample) — Cumulative PnL\n"
        f"Total: €{metrics['total_pnl_eur']:,.0f}  |  "
        f"Sharpe: {metrics['sharpe_ratio']}  |  "
        f"Win Rate: {metrics['win_rate_pct']}%  |  "
        f"Max Drawdown: €{metrics['max_drawdown_eur']:,.0f}"
    )
    ax1.set_ylabel("Cumulative PnL (EUR)")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df.index, df["price_eur_mwh"], color="gray", linewidth=0.8, alpha=0.7)
    buy_mask  = df["signal"] == 1
    sell_mask = df["signal"] == -1
    ax2.scatter(df.index[buy_mask],  df["price_eur_mwh"][buy_mask],
                color="green", s=2, label="Buy",  alpha=0.6)
    ax2.scatter(df.index[sell_mask], df["price_eur_mwh"][sell_mask],
                color="red",   s=2, label="Sell", alpha=0.6)
    ax2.set_ylabel("Price (EUR/MWh)")
    ax2.set_xlabel("Date")
    ax2.legend(markerscale=4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"backtest_{zone}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Backtest plot saved to {out}")


def run_backtest(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    n_hours: int = 6,
    zone: str = "DK1",
) -> Tuple[pd.DataFrame, dict]:
    """Run the full backtest pipeline on out-of-sample data only."""
    print(f"  Generating signals ({n_hours} buy + {n_hours} sell hours per day)...")
    df = generate_signals(df, model, n_hours=n_hours)

    print(f"  Calculating PnL...")
    df = calculate_pnl(df)

    metrics = calculate_metrics(df)

    print(f"\n  --- Backtest Results: {zone} (out-of-sample) ---")
    print(f"  Total PnL:       €{metrics['total_pnl_eur']:,.2f}")
    print(f"  Avg daily PnL:   €{metrics['avg_daily_pnl_eur']:,.2f}")
    print(f"  Win rate:        {metrics['win_rate_pct']}%")
    print(f"  Sharpe ratio:    {metrics['sharpe_ratio']}")
    print(f"  Max drawdown:    €{metrics['max_drawdown_eur']:,.2f}")
    print(f"  Trades:          {metrics['n_trades']}")

    plot_backtest(df, metrics, zone=zone)
    return df, metrics


if __name__ == "__main__":
    from features import load_processed
    from model import train_model

    for zone in ["DK1", "DK2"]:
        print(f"\n{'='*50}")
        print(f"Running backtest for {zone}")
        print(f"{'='*50}")

        df = load_processed(zone=zone)

        # Strict chronological split — train on first 75%, test on last 25%
        print("\nSplitting data (75% train / 25% test)...")
        train_df, test_df = train_test_split(df, train_ratio=0.75)

        # Train ONLY on training data
        print("\nTraining model on training set only...")
        model = train_model(train_df)

        # Backtest ONLY on unseen test data
        print("\nRunning backtest on unseen test data...")
        results, metrics = run_backtest(test_df, model, n_hours=6, zone=zone)

    print("\nDone! Check the models/ folder for backtest plots.")