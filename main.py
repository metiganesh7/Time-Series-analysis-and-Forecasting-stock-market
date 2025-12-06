import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.utils import download_data, prepare_series, train_test_split_series
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ----------------------------
# PLOT FUNCTION
# ----------------------------
def plot_series(train, test, preds, title, path):
    plt.figure(figsize=(10,4))
    train.plot(label='Train')
    test.plot(label='Test')
    preds.plot(label='Forecast')
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main(ticker, start, end):
    # 1. Load Data
    df = download_data(ticker, start=start, end=end, save_path='data')
    series = prepare_series(df, col='Close', freq='D')
    series.name = 'Close'

    train, test = train_test_split_series(series, test_size=0.2)

    # ----------------------------
    # SARIMA
    # ----------------------------
    sarima_res = train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12), save_path='models')
    sarima_pred = forecast_sarima(sarima_res, steps=len(test))
    sarima_pred = pd.Series(sarima_pred, index=test.index)
    plot_series(train, test, sarima_pred, 'SARIMA Forecast', os.path.join('plots', 'sarima.png'))

    # ----------------------------
    # PROPHET
    # ----------------------------
    prophet_model = train_prophet(train, save_path='models')
    prophet_pred = forecast_prophet(prophet_model, periods=len(test))
    prophet_pred = pd.Series(prophet_pred.values, index=test.index)
    plot_series(train, test, prophet_pred, 'Prophet Forecast', os.path.join('plots', 'prophet.png'))

    # ----------------------------
    # LSTM
    # ----------------------------
    scaler = MinMaxScaler()
    values = series.values.reshape(-1,1)
    scaler.fit(values)
    scaled = scaler.transform(values)

    split = int(len(scaled)*0.8)
    train_s = scaled[:split]
    test_s = scaled[split:]

    lstm_model = train_lstm(train_s, seq_len=60, epochs=5, batch_size=32, save_path='models')
    lstm_pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=60, steps=len(test))
    lstm_pred = pd.Series(lstm_pred, index=test.index)
    plot_series(train, test, lstm_pred, 'LSTM Forecast', os.path.join('plots', 'lstm.png'))

    st.success("Model training & forecasting completed. Check plots/ and models/ folders.")

# ----------------------------
# SCRIPT ENTRY POINT
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='ADANIPORTS.NS')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default=None)
    args = parser.parse_args()

    main(args.ticker, args.start, args.end)
