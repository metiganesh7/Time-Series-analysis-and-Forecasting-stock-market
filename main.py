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
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=3600)
def cached_download(ticker, start, end):
    df = download_data(ticker, start=start, end=end, save_path='data')
    return df

def detect_date_column(df):
    date_candidates = ["Date", "date", "datetime", "Datetime", "Timestamp", "timestamp"]
    for col in df.columns:
        if col in date_candidates or "date" in col.lower():
            return col
    return None

def detect_price_column(df):
    candidates = ["Close", "close", "CLOSE", "Adj Close", "Adj_Close", "Price", "price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include="number").columns
    return numeric_cols[-1] if len(numeric_cols) else None

def ensure_series(obj):
    if isinstance(obj, pd.DataFrame):
        price_col = detect_price_column(obj)
        return obj[price_col]
    return obj

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

# -------------------------
# SIDEBAR — DATA SOURCE
# -------------------------
st.sidebar.title("Data Source")

data_source = st.sidebar.radio(
    "Select data source:",
    ["YFinance", "Local CSV (from /data folder)", "Upload CSV"]
)

df = None

# --- Case 1: YFinance ---
if data_source == "YFinance":
    ticker = st.sidebar.text_input("Ticker (YFinance)", value="ADANIPORTS.NS")
    start = st.sidebar.date_input("Start date", value=dt.date(2015,1,1))
    end = st.sidebar.date_input("End date", value=dt.date.today())

    df = cached_download(ticker, start.isoformat(), end.isoformat())
    st.success(f"Loaded YFinance data for {ticker}")

# --- Case 2: Local CSV ---
elif data_source == "Local CSV (from /data folder)":
    local_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    selected = st.sidebar.selectbox("Choose dataset", local_files)

    df = pd.read_csv(os.path.join("data", selected))
    st.success(f"Loaded local dataset: {selected}")

# --- Case 3: Upload CSV ---
elif data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded successfully.")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# -------------------------
# CLEAN & PREPARE DF
# -------------------------
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
if date_col is None:
    st.error("No date column found in your dataset. A 'Date' column is required.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
if price_col is None:
    st.error("No price column found in dataset.")
    st.stop()

series = df[price_col]
series.name = price_col

# -------------------------
# DATA PREVIEW
# -------------------------
with st.expander("📊 Data Preview", expanded=True):
    st.write(df.tail())

    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax)
    ax.set_title(f"Price Series ({price_col})")
    st.pyplot(fig)
    plt.close(fig)

# Prepare series for models
series = prepare_series(df, col=price_col, freq='D')
train, test = train_test_split_series(series, test_size=0.2)

# -------------------------
# SIDEBAR — MODEL SELECTION
# -------------------------
st.sidebar.markdown("---")
model_options = st.sidebar.multiselect(
    "Models to run",
    ["SARIMA", "ARIMA", "Prophet", "LSTM"],
    default=["SARIMA", "Prophet", "LSTM"]
)

train_button = st.sidebar.button("Run Models")

# Advanced SARIMA / ARIMA / LSTM params
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(',')))
sarima_seasonal = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(',')))
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(',')))

lstm_seq = st.sidebar.number_input("LSTM seq length", 10, 365, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)

# -------------------------
# MODEL RUN FUNCTIONS
# -------------------------
col_left, col_right = st.columns(2)

def run_arima_and_show():
    if not HAVE_ARIMA:
        st.error("ARIMA model not available.")
        return

    train_series = train if isinstance(train, pd.Series) else train.iloc[:,0]
    arima_res = train_arima(train_series, order=arima_order)
    arima_pred = forecast_arima(arima_res, steps=len(test))
    arima_pred = pd.Series(arima_pred, index=test.index)

    buf = plot_series_inline(train, test, arima_pred, "ARIMA Forecast")
    save_plot_buf(buf, "plots/arima.png")
    col_right.subheader("ARIMA Forecast")
    col_right.image(buf)

def run_sarima_and_show():
    sarima_res = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal)
    sarima_pred = forecast_sarima(sarima_res, steps=len(test))
    sarima_pred = pd.Series(sarima_pred, index=test.index)

    buf = plot_series_inline(train, test, sarima_pred, "SARIMA Forecast")
    save_plot_buf(buf, "plots/sarima.png")
    col_left.subheader("SARIMA Forecast")
    col_left.image(buf)

def run_prophet_and_show():
    prophet_model = train_prophet(train)
    prophet_pred = forecast_prophet(prophet_model, periods=len(test))
    prophet_pred = pd.Series(prophet_pred.values, index=test.index)

    buf = plot_series_inline(train, test, prophet_pred, "Prophet Forecast")
    save_plot_buf(buf, "plots/prophet.png")
    col_left.subheader("Prophet Forecast")
    col_left.image(buf)

def run_lstm_and_show():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    split = int(len(scaled)*0.8)
    train_s = scaled[:split]

    lstm_model = train_lstm(train_s, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)

    lstm_pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
    lstm_pred = pd.Series(lstm_pred, index=test.index)

    buf = plot_series_inline(train, test, lstm_pred, "LSTM Forecast")
    save_plot_buf(buf, "plots/lstm.png")
    col_right.subheader("LSTM Forecast")
    col_right.image(buf)

# -------------------------
# RUN SELECTED MODELS
# -------------------------
if train_button:
    if "SARIMA" in model_options:
        run_sarima_and_show()

    if "ARIMA" in model_options:
        run_arima_and_show()

    if "Prophet" in model_options:
        run_prophet_and_show()

    if "LSTM" in model_options:
        run_lstm_and_show()
