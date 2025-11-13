
# Stock Market Forecasting Project

This repository implements time-series forecasting for stock prices using ARIMA, SARIMA, Prophet, and LSTM models.

## Contents
- `data/` : place to store datasets (scripts download data automatically)
- `notebooks/` : Jupyter notebook with end-to-end code
- `scripts/` : Python scripts for each model and utilities
- `models/` : saved model files (after running)
- `plots/` : generated plots (after running)
- `main.py` : orchestrator script to run the pipeline
- `requirements.txt` : Python dependencies

## How to run
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```
2. To run the full pipeline (downloads data, trains models):
   ```bash
   python main.py --ticker "ADANIPORTS.NS" --start "2015-01-01" --end "2024-12-31"
   ```
3. Check `plots/` and `models/` for outputs.

Note: This project uses `yfinance` to download historical stock data. Prophet requires `prophet` package.
