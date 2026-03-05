"""
model.py
--------
XGBoost forecasting model for day-ahead electricity prices.
Uses time-series cross-validation to evaluate performance honestly —
no random train/test splits, which would leak future data into the past.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS_DIR = Path(__file__).parent.parent / "models"

# Features used for training — everything except the target
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_monday",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "rolling_mean_24h", "rolling_mean_168h", "rolling_std_24h",
    "rolling_min_24h", "rolling_max_24h", "hist_avg_by_hour",
]

TARGET_COL = "price_eur_mwh"


def time_series_split(df: pd.DataFrame, n_splits: int = 5):
    """
    Generate time-series cross-validation folds.
    Each fold uses all past data for training and the next chunk for validation.
    This respects temporal order — we never train on future data.
    """
    n = len(df)
    fold_size = n // (n_splits + 1)

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end   = fold_size * (i + 1)
        train = df.iloc[:train_end]
        val   = df.iloc[train_end:val_end]
        yield train, val


def train_model(
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost model on the full dataset.

    Args:
        df:           Feature DataFrame from features.py
        feature_cols: List of feature column names
        target_col:   Name of the target column

    Returns:
        Trained XGBRegressor
    """
    X = df[feature_cols]
    y = df[target_col]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    model.fit(X, y)
    print(f"  Model trained on {len(X)} samples with {len(feature_cols)} features.")
    return model


def cross_validate(
    df: pd.DataFrame,
    n_splits: int = 5,
    feature_cols: list = FEATURE_COLS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Evaluate model performance using time-series cross-validation.

    Returns:
        DataFrame with MAE and RMSE for each fold
    """
    results = []

    for fold, (train, val) in enumerate(time_series_split(df, n_splits), 1):
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_val   = val[feature_cols]
        y_val   = val[target_col]

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mean_price = y_val.mean()

        results.append({
            "fold":        fold,
            "train_size":  len(train),
            "val_size":    len(val),
            "mae":         round(mae, 2),
            "rmse":        round(rmse, 2),
            "mean_price":  round(mean_price, 2),
            "mae_pct":     round(mae / mean_price * 100, 1),  # MAE as % of mean price
        })

        print(f"  Fold {fold}: MAE={mae:.2f} EUR/MWh  RMSE={rmse:.2f}  "
              f"MAE%={mae/mean_price*100:.1f}%  (val size={len(val)})")

    return pd.DataFrame(results)


def plot_predictions(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    n_days: int = 14,
    feature_cols: list = FEATURE_COLS,
    target_col: str = TARGET_COL,
    zone: str = "DK1",
):
    """
    Plot predicted vs actual prices for the last n_days of data.
    Saves the plot to models/
    """
    sample = df.tail(n_days * 24)
    X = sample[feature_cols]
    y = sample[target_col]
    preds = model.predict(X)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample.index, y.values,    label="Actual",    linewidth=1.5, color="steelblue")
    ax.plot(sample.index, preds,       label="Predicted", linewidth=1.5, color="orange", linestyle="--")
    ax.set_title(f"{zone} Day-Ahead Price Forecast — Last {n_days} Days")
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"predictions_{zone}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved to {out}")


def plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_cols: list = FEATURE_COLS,
    zone: str = "DK1",
):
    """Plot and save feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{zone} Feature Importance")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    out = MODELS_DIR / f"feature_importance_{zone}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved to {out}")


if __name__ == "__main__":
    from features import load_processed

    for zone in ["DK1", "DK2"]:
        print(f"\n{'='*50}")
        print(f"Training model for {zone}")
        print(f"{'='*50}")

        df = load_processed(zone=zone)

        print("\nRunning time-series cross-validation...")
        cv_results = cross_validate(df, n_splits=5)
        print(f"\nMean MAE:  {cv_results['mae'].mean():.2f} EUR/MWh")
        print(f"Mean RMSE: {cv_results['rmse'].mean():.2f} EUR/MWh")
        print(f"Mean MAE%: {cv_results['mae_pct'].mean():.1f}% of mean price")

        print("\nTraining final model on full dataset...")
        model = train_model(df)

        print("\nGenerating plots...")
        plot_predictions(model, df, n_days=14, zone=zone)
        plot_feature_importance(model, zone=zone)

    print("\nDone! Check the models/ folder for output plots.")