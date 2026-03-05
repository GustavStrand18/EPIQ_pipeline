"""
run_pipeline.py
---------------
Runs the full electricity price forecasting pipeline end to end.

Usage:
    python run_pipeline.py             # Run full pipeline for DK1 and DK2
    python run_pipeline.py --zone DK1  # Run for a single zone only
    python run_pipeline.py --skip-fetch  # Skip data fetch (use existing raw data)

Steps:
    1. Fetch raw day-ahead prices from Energi Data Service API
    2. Run data quality checks
    3. Build feature matrix
    4. Train XGBoost forecasting model with cross-validation
    5. Run out-of-sample backtest
"""

import argparse
import sys
import time
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import fetch_day_ahead_prices, save_raw, load_raw
from data_quality import run_quality_checks
from features import build_features, save_processed, load_processed
from model import train_model, cross_validate, plot_predictions, plot_feature_importance
from backtest import train_test_split, run_backtest


def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def run_zone(zone: str, skip_fetch: bool, start: str, end: str):
    separator(f"Zone: {zone}")
    t0 = time.time()

    # --- Step 1: Data Ingestion ---
    print(f"\n[1/5] Data Ingestion")
    if skip_fetch:
        print(f"  Skipping fetch — loading existing raw data...")
        df_raw = load_raw(zone=zone)
        print(f"  Loaded {len(df_raw)} rows.")
    else:
        df_raw = fetch_day_ahead_prices(zone=zone, start=start, end=end)
        save_raw(df_raw, zone=zone)

    # --- Step 2: Data Quality ---
    print(f"\n[2/5] Data Quality Checks")
    report = run_quality_checks(df_raw, zone=zone)
    if not report.passed:
        print(f"\n  WARNING: Data quality checks failed for {zone}.")
        print(f"  Missing values: {report.missing_values}")
        print(f"  Time gaps: {len(report.time_gaps)}")
        print(f"  Continuing anyway — check the data before using in production.")

    # --- Step 3: Feature Engineering ---
    print(f"\n[3/5] Feature Engineering")
    df_features = build_features(df_raw)
    save_processed(df_features, zone=zone)

    # --- Step 4: Model Training & Cross-Validation ---
    print(f"\n[4/5] Model Training")
    print(f"  Running 5-fold time-series cross-validation...")
    cv_results = cross_validate(df_features, n_splits=5)
    print(f"\n  CV Summary:")
    print(f"    Mean MAE:  {cv_results['mae'].mean():.2f} EUR/MWh")
    print(f"    Mean RMSE: {cv_results['rmse'].mean():.2f} EUR/MWh")
    print(f"    Mean MAE%: {cv_results['mae_pct'].mean():.1f}% of mean price")

    print(f"\n  Training final model on full dataset...")
    model = train_model(df_features)
    plot_predictions(model, df_features, n_days=14, zone=zone)
    plot_feature_importance(model, zone=zone)

    # --- Step 5: Backtest ---
    print(f"\n[5/5] Backtesting (out-of-sample)")
    train_df, test_df = train_test_split(df_features, train_ratio=0.75)
    backtest_model = train_model(train_df)
    results, metrics = run_backtest(test_df, backtest_model, n_hours=6, zone=zone)

    elapsed = time.time() - t0
    print(f"\n  Completed {zone} in {elapsed:.1f}s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run the electricity price forecasting pipeline.")
    parser.add_argument("--zone", choices=["DK1", "DK2", "both"], default="both",
                        help="Bidding zone to run (default: both)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip data fetch and use existing raw CSV files")
    parser.add_argument("--start", default="2022-01-01",
                        help="Start date for data fetch (default: 2022-01-01)")
    parser.add_argument("--end", default="2024-01-01",
                        help="End date for data fetch (default: 2024-01-01)")
    args = parser.parse_args()

    zones = ["DK1", "DK2"] if args.zone == "both" else [args.zone]

    print("=" * 55)
    print("  Electricity Price Forecasting Pipeline")
    print("=" * 55)
    print(f"  Zones:       {', '.join(zones)}")
    print(f"  Period:      {args.start} → {args.end}")
    print(f"  Skip fetch:  {args.skip_fetch}")

    all_metrics = {}
    for zone in zones:
        all_metrics[zone] = run_zone(
            zone=zone,
            skip_fetch=args.skip_fetch,
            start=args.start,
            end=args.end,
        )

    # Final summary
    separator("Pipeline Complete — Summary")
    for zone, metrics in all_metrics.items():
        print(f"\n  {zone}:")
        print(f"    Total PnL:     €{metrics['total_pnl_eur']:>10,.2f}")
        print(f"    Avg daily PnL: €{metrics['avg_daily_pnl_eur']:>10,.2f}")
        print(f"    Win rate:       {metrics['win_rate_pct']:>9.1f}%")
        print(f"    Sharpe ratio:   {metrics['sharpe_ratio']:>9.2f}")
        print(f"    Max drawdown:  €{metrics['max_drawdown_eur']:>10,.2f}")

    print(f"\n  Output plots saved to: models/")
    print(f"  Processed data saved to: data/processed/")


if __name__ == "__main__":
    main()
