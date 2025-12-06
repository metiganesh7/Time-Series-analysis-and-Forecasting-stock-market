import sys
import os
import io
import joblib
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# project model utilities
from scripts.utils import download_data, prepare_series, train_test_split_series

# Models
try:
    from scripts.arima_model import train_arima, forecast_arima
    HAVE_ARIMA = True
except Exception:
    HAVE_ARIMA = False

from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# plotting & UI libs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# create directories if missing
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=3600)
def cached_download(ticker, start, end):
    df = download_data(ticker, start=start, end=end, save_path='data')
    return df

def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def plot_series_inline(train, test, preds, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label='Train')
    test.plot(ax=ax, label='Test')
    preds.plot(ax=ax, label='Forecast')
    ax.set_title(title)
    ax.legend()
    buf = plot_to_image(fig)
    plt.close(fig)
    return buf

def save_plot_buf(buf, filename):
    with open(filename, 'wb') as f:
        f.write(buf.getbuffer())

def ensure_series(obj):
    if isinstance(obj, pd.DataFrame):
        if 'Close' in obj.columns:
            return obj['Close']
        return obj.iloc[:, 0]
    return obj

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Ticker (yfinance)", value="ADANIPORTS.NS")
start = st.sidebar.date_input("Start date", value=dt.date(2015,1,1))
end = st.sidebar.date_input("End date", value=dt.date.today())

model_options = st.sidebar.multiselect("Models to run", ["SARIMA", "ARIMA", "Prophet", "LSTM"],
                                       default=["SARIMA", "Prophet", "LSTM"])
train_button = st.sidebar.button("Run Models")

st.sidebar.markdown("---")
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA order (p,d,q)", value="1,1,1").split(',')))
sarima_seasonal = tuple(map(int, st.sidebar.text_input("SARIMA seasonal_order (P,D,Q,s)", value="1,1,1,12").split(',')))
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA order (p,d,q)", value="5,1,0").split(',')))
lstm_seq = st.sidebar.number_input("LSTM sequence length", min_value=10, max_value=365, value=60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", min_value=1, max_value=50, value=5)
lstm_batch = st.sidebar.number_input("LSTM batch size", min_value=1, max_value=256, value=32)

# -------------------------
# Main layout
# -------------------------
st.title("📈 Time Series Forecasting Dashboard")

# Load data
with st.expander("Data Preview", expanded=True):
    df = cached_download(ticker, start.isoformat(), end.isoformat())
    st.dataframe(df.tail())

    fig, ax = plt.subplots(figsize=(10,3))
    ensure_series(df).plot(ax=ax)
    ax.set_title(f"{ticker} Close Price")
    st.pyplot(fig)
    plt.close(fig)

# Prepare series
series = prepare_series(df, col='Close', freq='D')
series = ensure_series(series)
series.name = 'Close'

train, test = train_test_split_series(series, test_size=0.2)

col_left, col_right = st.columns(2)

# -----------------------------------------------------------
# FIXED ARIMA HANDLER (No more 'Close' errors)
# -----------------------------------------------------------
def run_arima_and_show():
    if not HAVE_ARIMA:
        st.warning("ARIMA module not found.")
        return

    st.info("Training ARIMA...")
    train_series = train if isinstance(train, pd.Series) else train.iloc[:,0]

    arima_res = train_arima(train_series, order=arima_order, save_path='models', name='arima_model.pkl')

    st.info("Forecasting ARIMA...")
    arima_pred = forecast_arima(arima_res, steps=len(test))
    arima_pred = pd.Series(arima_pred, index=test.index)

    buf = plot_series_inline(train, test, arima_pred, 'ARIMA Forecast')
    save_plot_buf(buf, "plots/arima.png")

    col_right.subheader("ARIMA Forecast")
    col_right.image(buf)

# -----------------------------------------------------------
# FIXED PROPHET HANDLER (1-D array issue resolved)
# -----------------------------------------------------------
def run_prophet_and_show():
    st.info("Training Prophet...")

    prophet_model = train_prophet(train, save_path='models', name='prophet_model.pkl')

    st.info("Forecasting Prophet...")
    prophet_pred = forecast_prophet(prophet_model, periods=len(test))
    prophet_pred = pd.Series(prophet_pred.values, index=test.index)

    buf = plot_series_inline(train, test, prophet_pred, 'Prophet Forecast')
    save_plot_buf(buf, "plots/prophet.png")

    col_left.subheader("Prophet Forecast")
    col_left.image(buf)

# -----------------------------------------------------------
# SARIMA
# -----------------------------------------------------------
def run_sarima_and_show():
    st.info("Training SARIMA...")
    sarima_res = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal, save_path="models")

    st.info("Forecasting SARIMA...")
    sarima_pred = forecast_sarima(sarima_res, steps=len(test))
    sarima_pred = pd.Series(sarima_pred, index=test.index)

    buf = plot_series_inline(train, test, sarima_pred, 'SARIMA Forecast')
    save_plot_buf(buf, "plots/sarima.png")

    col_left.subheader("SARIMA Forecast")
    col_left.image(buf)

# -----------------------------------------------------------
# LSTM
# -----------------------------------------------------------
def run_lstm_and_show():
    st.info("Training LSTM...")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    split_idx = int(len(scaled) * 0.8)
    train_s = scaled[:split_idx]

    lstm_model = train_lstm(train_s, seq_len=lstm_seq, epochs=int(lstm_epochs),
                            batch_size=int(lstm_batch), save_path='models', name='lstm_model.h5')

    st.info("Forecasting LSTM...")
    lstm_pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
    lstm_pred = pd.Series(lstm_pred, index=test.index)

    buf = plot_series_inline(train, test, lstm_pred, 'LSTM Forecast')
    save_plot_buf(buf, "plots/lstm.png")

    col_right.subheader("LSTM Forecast")
    col_right.image(buf)

# -----------------------------------------------------------
# RUN SELECTED MODELS
# -----------------------------------------------------------
if train_button:
    if "SARIMA" in model_options:
        run_sarima_and_show()

    if "ARIMA" in model_options:
        try:
            run_arima_and_show()
        except Exception as e:
            st.error(f"ARIMA failed: {e}")

    if "Prophet" in model_options:
        try:
            run_prophet_and_show()
        except Exception as e:
            st.error(f"Prophet failed: {e}")

    if "LSTM" in model_options:
        try:
            run_lstm_and_show()
        except Exception as e:
            st.error(f"LSTM failed: {e}")
