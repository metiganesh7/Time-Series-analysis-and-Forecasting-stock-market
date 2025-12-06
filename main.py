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
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]  # fallback

def detect_price_column(df):
    candidates = ["Close", "close", "Price", "Adj Close", "Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    return numeric[-1]  # fallback

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

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")
st.title("📈 Time Series Forecasting Dashboard (Upload CSV)")

st.sidebar.header("📁 Upload CSV File")
uploaded = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if uploaded is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

# Detect columns
date_col = detect_date_column(df)
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
series = df[price_col]
series.name = price_col

# Preview
with st.expander("📊 Data Preview", expanded=True):
    st.dataframe(df.tail())

    fig, ax = plt.subplots(figsize=(10, 3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

# Prepare daily aggregated series
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)


# -------------------------------------------------
# Sidebar — Model Settings
# -------------------------------------------------

st.sidebar.header("🧠 Models")
models_to_run = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
)

st.sidebar.markdown("---")

# ARIMA
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))

# SARIMA
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
sarima_seasonal = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

# LSTM
lstm_seq = st.sidebar.number_input("LSTM Sequence Length", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM Batch Size", 1, 256, 32)

run_btn = st.sidebar.button("Run Models")

col1, col2 = st.columns(2)


# -------------------------------------------------
# Model Execution Functions
# -------------------------------------------------

def run_arima_forecast():
    model = train_arima(train.squeeze(), order=arima_order)
    pred = forecast_arima(model, len(test))
    return pd.Series(pred, index=test.index)

def run_sarima_forecast():
    model = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=sarima_seasonal)
    pred = forecast_sarima(model, len(test))
    return pd.Series(pred, index=test.index)

def run_prophet_forecast():
    series_for_prophet = train.squeeze()
    model = train_prophet(series_for_prophet)
    pred = forecast_prophet(model, len(test))
    return pred.reindex(test.index)  # align with test index

def run_lstm_forecast():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    split = int(len(scaled)*0.8)
    train_scaled = scaled[:split]
    lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
    pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
    return pd.Series(pred, index=test.index)


# -------------------------------------------------
# Run Models
# -------------------------------------------------

if run_btn:

    # ARIMA
    if "ARIMA" in models_to_run:
        with st.spinner("Running ARIMA..."):
            pred = run_arima_forecast()
            buf = plot_series(train, test, pred, "ARIMA Forecast")
            col1.subheader("ARIMA Forecast")
            col1.image(buf)

    # SARIMA
    if "SARIMA" in models_to_run:
        with st.spinner("Running SARIMA..."):
            pred = run_sarima_forecast()
            buf = plot_series(train, test, pred, "SARIMA Forecast")
            col1.subheader("SARIMA Forecast")
            col1.image(buf)

    # Prophet
    if "Prophet" in models_to_run:
        with st.spinner("Running Prophet..."):
            pred = run_prophet_forecast()
            buf = plot_series(train, test, pred, "Prophet Forecast")
            col2.subheader("Prophet Forecast")
            col2.image(buf)

    # LSTM
    if "LSTM" in models_to_run:
        with st.spinner("Running LSTM..."):
            pred = run_lstm_forecast()
            buf = plot_series(train, test, pred, "LSTM Forecast")
            col2.subheader("LSTM Forecast")
            col2.image(buf)
