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

# auto create folders
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data(ttl=3600)
def cached_download(ticker, start, end):
    return download_data(ticker, start, end, "data")

def detect_date_column(df):
    candidates = ["Date", "date", "Datetime", "datetime", "Timestamp", "timestamp"]
    for col in df.columns:
        if col in candidates or "date" in col.lower():
            return col
    return None

def detect_price_column(df):
    candidates = ["Close", "close", "CLOSE", "Adj Close", "Adj_Close", "Price", "price"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback → last numeric column
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
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    preds.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    buf = plot_to_image(fig)
    plt.close(fig)
    return buf

def save_plot_buf(buf, filename):
    with open(filename, "wb") as f:
        f.write(buf.getbuffer())


# -------------------------
# SIDEBAR — DATA SOURCE
# -------------------------
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["YFinance", "Upload CSV", "Enter CSV Path"]
)

df = None

# ——— CASE 1: YFINANCE ———
if data_source == "YFinance":
    ticker = st.sidebar.text_input("Ticker", "ADANIPORTS.NS")
    start = st.sidebar.date_input("Start date", dt.date(2015,1,1))
    end = st.sidebar.date_input("End date", dt.date.today())

    df = cached_download(ticker, start.isoformat(), end.isoformat())
    st.success(f"Loaded YFinance dataset for {ticker}")

# ——— CASE 2: Upload CSV ———
elif data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded successfully.")
    else:
        st.info("Upload a CSV file to continue.")
        st.stop()

# ——— CASE 3: Enter CSV Path ———
elif data_source == "Enter CSV Path":
    path = st.sidebar.text_input("Full CSV path", "")
    if path.strip():
        try:
            df = pd.read_csv(path)
            st.success(f"Loaded file: {path}")
        except Exception as e:
            st.error(f"Could not load file: {e}")
            st.stop()
    else:
        st.info("Enter a valid file path.")
        st.stop()


# -------------------------
# STOP if df is STILL None
# -------------------------
if df is None:
    st.error("No dataset loaded. Please select a data source.")
    st.stop()

# -------------------------
# CLEAN DATAFRAME
# -------------------------
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
if date_col is None:
    st.error("Could not determine the Date column in your dataset.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
if price_col is None:
    st.error("Could not determine price column. Must contain Close / Price column.")
    st.stop()

series = df[price_col]
series.name = price_col

# -------------------------
# PREVIEW SECTION
# -------------------------
with st.expander("📊 Dataset Preview", expanded=True):
    st.dataframe(df.tail())

    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax)
    ax.set_title(f"Price Series — {price_col}")
    st.pyplot(fig)
    plt.close(fig)

# -------------------------
# Prepare series for models
# -------------------------
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)

# -------------------------
# SIDEBAR — MODEL SELECTION
# -------------------------
st.sidebar.header("🧠 Models")

model_options = st.sidebar.multiselect(
    "Select Models",
    ["SARIMA", "ARIMA", "Prophet", "LSTM"],
    default=["SARIMA", "Prophet", "LSTM"]
)

run_btn = st.sidebar.button("Run Models")

# Hyperparameters
st.sidebar.markdown("---")
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
sarima_seasonal = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))

lstm_seq = st.sidebar.number_input("LSTM Sequence Length", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM Batch Size", 1, 256, 32)

col_left, col_right = st.columns(2)


# -------------------------
# MODEL RUN FUNCTIONS
# -------------------------
def run_arima():
    train_series = train if isinstance(train, pd.Series) else train.iloc[:,0]
    model = train_arima(train_series, order=arima_order)
    pred = forecast_arima(model, steps=len(test))
    return pd.Series(pred, index=test.index)

def run_sarima():
    model = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal)
    pred = forecast_sarima(model, steps=len(test))
    return pd.Series(pred, index=test.index)

def run_prophet_model():
    model = train_prophet(train)
    pred = forecast_prophet(model, periods=len(test))
    return pd.Series(pred.values, index=test.index)

def run_lstm_model():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    split = int(len(scaled)*0.8)
    train_s = scaled[:split]

    model = train_lstm(train_s, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
    pred = forecast_lstm(model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
    return pd.Series(pred, index=test.index)


# -------------------------
# RUN MODELS
# -------------------------
if run_btn:
    if "ARIMA" in model_options:
        if HAVE_ARIMA:
            with st.spinner("Running ARIMA..."):
                pred = run_arima()
                buf = plot_series_inline(train, test, pred, "ARIMA Forecast")
                col_left.subheader("ARIMA Forecast")
                col_left.image(buf)
        else:
            st.error("ARIMA model script missing.")

    if "SARIMA" in model_options:
        with st.spinner("Running SARIMA..."):
            pred = run_sarima()
            buf = plot_series_inline(train, test, pred, "SARIMA Forecast")
            col_left.subheader("SARIMA Forecast")
            col_left.image(buf)

    if "Prophet" in model_options:
        with st.spinner("Running Prophet..."):
            pred = run_prophet_model()
            buf = plot_series_inline(train, test, pred, "Prophet Forecast")
            col_right.subheader("Prophet Forecast")
            col_right.image(buf)

    if "LSTM" in model_options:
        with st.spinner("Running LSTM..."):
            pred = run_lstm_model()
            buf = plot_series_inline(train, test, pred, "LSTM Forecast")
            col_right.subheader("LSTM Forecast")
            col_right.image(buf)
