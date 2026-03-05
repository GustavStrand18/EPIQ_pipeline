"""
data_ingestion.py
-----------------
Fetches day-ahead electricity prices from Energi Data Service
(api.energidataservice.dk) for Danish bidding zones DK1 and DK2.

No API key required — this is a fully open Danish government API.
"""

import requests
import pandas as pd
from pathlib import Path
from typing import Literal

VALID_ZONES = ["DK1", "DK2"]
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE_URL = "https://api.energidataservice.dk/dataset/Elspotprices"


def fetch_day_ahead_prices(
    zone: Literal["DK1", "DK2"] = "DK1",
    start: str = "2022-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Fetch day-ahead electricity prices for a Danish bidding zone.

    Args:
        zone:  'DK1' (West Denmark) or 'DK2' (East Denmark)
        start: Start date as string 'YYYY-MM-DD'
        end:   End date as string 'YYYY-MM-DD'

    Returns:
        DataFrame with timestamp index and price_eur_mwh, price_dkk_mwh columns
    """
    if zone not in VALID_ZONES:
        raise ValueError(f"Unknown zone '{zone}'. Choose from {VALID_ZONES}")

    print(f"Fetching day-ahead prices for {zone} from {start} to {end}...")

    all_records = []
    offset = 0
    limit = 10_000  # Max records per request

    while True:
        params = {
            "start": f"{start}T00:00",
            "end": f"{end}T00:00",
            "filter": f'{{"PriceArea":"{zone}"}}',
            "sort": "HourUTC ASC",
            "limit": limit,
            "offset": offset,
        }

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        all_records.extend(records)

        total = data.get("total", 0)
        offset += limit

        print(f"  Fetched {len(all_records)} / {total} records...")

        if offset >= total:
            break

    if not all_records:
        raise ValueError(f"No data returned for zone {zone} between {start} and {end}.")

    df = pd.DataFrame(all_records)

    # Parse timestamps and set as index
    df["timestamp"] = pd.to_datetime(df["HourUTC"], utc=True)
    df = df.set_index("timestamp")

    # Keep and rename relevant columns
    df = df[["SpotPriceEUR", "SpotPriceDKK"]].rename(
        columns={
            "SpotPriceEUR": "price_eur_mwh",
            "SpotPriceDKK": "price_dkk_mwh",
        }
    )

    df = df.sort_index()
    print(f"  Done. {len(df)} hourly records fetched.")
    return df


def save_raw(df: pd.DataFrame, zone: str) -> Path:
    """Save raw price data to CSV in data/raw/"""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = RAW_DATA_DIR / f"day_ahead_prices_{zone}.csv"
    df.to_csv(filename)
    print(f"  Saved to {filename}")
    return filename


def load_raw(zone: str = "DK1") -> pd.DataFrame:
    """Load previously saved raw price data from CSV."""
    filename = RAW_DATA_DIR / f"day_ahead_prices_{zone}.csv"
    if not filename.exists():
        raise FileNotFoundError(
            f"No raw data found for {zone}. Run fetch_day_ahead_prices() first."
        )
    df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
    return df


if __name__ == "__main__":
    for zone in ["DK1", "DK2"]:
        df = fetch_day_ahead_prices(zone=zone, start="2022-01-01", end="2024-01-01")
        save_raw(df, zone=zone)
        print(df.head())
        print()
