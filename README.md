# European Electricity Price Forecasting Pipeline

An end-to-end data pipeline and forecasting system for day-ahead electricity prices in Danish bidding zones (DK1/DK2), built on the ENTSO-E Transparency Platform.

## What it does

1. **Ingests** hourly day-ahead prices from ENTSO-E API
2. **Validates** data quality (missing values, time gaps, outliers)
3. **Engineers features** relevant to electricity price forecasting
4. **Trains** an XGBoost forecasting model with time-series cross-validation
5. **Backtests** a simple trading strategy based on price forecasts

## Project Structure

```
epiq_pipeline/
├── data/
│   ├── raw/            # Raw data from ENTSO-E (not committed to git)
│   └── processed/      # Cleaned and feature-engineered data
├── src/
│   ├── data_ingestion.py   # ENTSO-E API client and data fetching
│   ├── data_quality.py     # Data quality checks
│   ├── features.py         # Feature engineering
│   ├── model.py            # Forecasting model
│   └── backtest.py         # Trading backtest engine
├── tests/
│   └── test_data_quality.py
├── notebooks/          # Exploratory analysis
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/yourusername/epiq_pipeline.git
cd epiq_pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get an ENTSO-E API key**

Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/). Approval usually takes 1-2 days.

**3. Set up your environment**
```bash
cp .env.example .env
# Edit .env and add your ENTSOE_API_KEY
```

## Usage

**Fetch raw data**
```bash
python src/data_ingestion.py
```

**Run data quality checks**
```bash
python src/data_quality.py
```

**Run tests**
```bash
pytest tests/
```

## Data Source

[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — the official source for European electricity market data.

Bidding zones used:
- `DK1` — Denmark West (connected to Germany/Netherlands)
- `DK2` — Denmark East (connected to Sweden/Germany)
