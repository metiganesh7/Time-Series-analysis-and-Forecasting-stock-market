import sys
import os
import io
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model modules
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower() or col.lower() in ["timestamp", "datetime"]:
            return col
    return None

def detect_price_column(df):
    candidates = ["Close", "close", "Price", "Adj Close"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include="number").columns
    return numeric_cols[-1]

def plot_series(train, test, preds, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    preds.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    return buffer


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")
st.title("📈 Time Series Forecasting Dashboard (Upload CSV Only)")

# Upload CSV
st.sidebar.header("📁 Upload CSV File")
uploaded = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if not uploaded:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

# Detect columns
date_col = detect_date_column(df)
if not date_col:
    st.error("No date column found. Please include a column named Date/Datetime.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
series = df[price_col]
series.name = price_col

# Show preview
with st.expander("📊 Data Preview", expanded=True):
    st.dataframe(df.tail())
    fig, ax = plt.subplots(figsize=(10, 3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

# Prepare data
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)

# -------------------------------------------------
# Model Selection
# -------------------------------------------------

st.sidebar.header("🧠 Models")
models_to_run = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
)

st.sidebar.markdown("---")

# ARIMA settings
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))

# SARIMA settings
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
sarima_seasonal = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

# LSTM settings
lstm_seq = st.sidebar.number_input("LSTM Sequence Length", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM Batch Size", 1, 256, 32)

run_btn = st.sidebar.button("Run Models")

col1, col2 = st.columns(2)

# -------------------------------------------------
# Run Models
# -------------------------------------------------

if run_btn:

    # ARIMA
    if "ARIMA" in models_to_run:
        with st.spinner("Running ARIMA..."):
            arima_model = train_arima(train, order=arima_order)
            arima_pred = pd.Series(forecast_arima(arima_model, len(test)), index=test.index)
            buf = plot_series(train, test, arima_pred, "ARIMA Forecast")
            col1.subheader("ARIMA Forecast")
            col1.image(buf)

    # SARIMA
    if "SARIMA" in models_to_run:
        with st.spinner("Running SARIMA..."):
            sarima_model = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal)
            sarima_pred = pd.Series(forecast_sarima(sarima_model, len(test)), index=test.index)
            buf = plot_series(train, test, sarima_pred, "SARIMA Forecast")
            col1.subheader("SARIMA Forecast")
            col1.image(buf)

    # Prophet
    if "Prophet" in models_to_run:
        with st.spinner("Running Prophet..."):
            prophet_model = train_prophet(train)
            prophet_pred = forecast_prophet(prophet_model, periods=len(test))
            prophet_pred = pd.Series(prophet_pred.values, index=test.index)
            buf = plot_series(train, test, prophet_pred, "Prophet Forecast")
            col2.subheader("Prophet Forecast")
            col2.image(buf)

    # LSTM
    if "LSTM" in models_to_run:
        with st.spinner("Running LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            train_scaled = scaled[: int(len(scaled)*0.8) ]

            lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            lstm_pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
            lstm_pred = pd.Series(lstm_pred, index=test.index)

            buf = plot_series(train, test, lstm_pred, "LSTM Forecast")
            col2.subheader("LSTM Forecast")
            col2.image(buf)
