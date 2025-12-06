# app.py
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

# models (ARIMA optional; SARIMA, Prophet, LSTM are expected)
# arima_model may or may not exist depending on your repo; wrap imports
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

def load_model_if_exists(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            # for keras h5 or other formats, allow joblib failure
            return None
    return None

def ensure_series(obj):
    # convert DataFrame column to Series if needed
    if isinstance(obj, pd.DataFrame):
        # prefer 'Close' column if present
        if 'Close' in obj.columns:
            s = obj['Close']
        else:
            s = obj.iloc[:, 0]
        return s
    return obj

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Ticker (yfinance)", value="ADANIPORTS.NS")
col1, col2 = st.sidebar.columns(2)
start = st.sidebar.date_input("Start date", value=dt.date(2015,1,1), key="start")
end = st.sidebar.date_input("End date (None = today)", value=dt.date.today(), key="end")

if end is None:
    end = dt.date.today()

model_options = st.sidebar.multiselect("Models to run", ["SARIMA", "ARIMA (if available)", "Prophet", "LSTM"],
                                       default=["SARIMA", "Prophet", "LSTM"])
train_button = st.sidebar.button("Run Selected Models")
full_pipeline = st.sidebar.button("Run Full Pipeline (all models)")

# advanced params
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced params")
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
st.write("Load historical data, train models, and view forecasts. Models are saved to `/models` and plots to `/plots`.")

# Load data
with st.expander("Data & Preview", expanded=True):
    st.write(f"Loading {ticker} from {start} to {end} ...")
    try:
        df = cached_download(ticker, start.isoformat(), end.isoformat())
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    st.write("Data snapshot:")
    st.dataframe(df.tail(10))

    st.write("Close price plot:")
    fig, ax = plt.subplots(figsize=(10,3))
    if 'Close' in df.columns:
        df['Close'].plot(ax=ax)
    else:
        df.iloc[:,0].plot(ax=ax)
    ax.set_title(f"{ticker} Close Price")
    st.pyplot(fig)
    plt.close(fig)

# Prepare series
try:
    series = prepare_series(df, col='Close', freq='D')
    series = ensure_series(series)
    series.name = 'Close'
except Exception as e:
    st.error(f"Failed to prepare series: {e}")
    st.stop()

train, test = train_test_split_series(series, test_size=0.2)

# Model results placeholders
col_left, col_right = st.columns(2)

# Function to run SARIMA
def run_sarima_and_show():
    with st.spinner("Training SARIMA..."):
        sarima_res = train_sarima(train, order=sarima_order, seasonal_order=sarima_seasonal, save_path='models', name='sarima_model.pkl')
    with st.spinner("Forecasting SARIMA..."):
        sarima_pred = forecast_sarima(sarima_res, steps=len(test))
        sarima_pred = pd.Series(sarima_pred, index=test.index)
    buf = plot_series_inline(train, test, sarima_pred, 'SARIMA Forecast')
    save_plot_buf(buf, os.path.join('plots', 'sarima.png'))
    col_left.subheader("SARIMA Forecast")
    col_left.image(buf)
    col_left.markdown(f"Saved model: `models/sarima_model.pkl`")

# Function to run ARIMA if available
def run_arima_and_show():
    if not HAVE_ARIMA:
        st.warning("ARIMA module not found in repository.")
        return
    with st.spinner("Training ARIMA..."):
        # convert train to series for ARIMA
        arima_res = train_arima(train if isinstance(train, pd.Series) else train['Close'], order=arima_order, save_path='models', name='arima_model.pkl')
    with st.spinner("Forecasting ARIMA..."):
        arima_pred = forecast_arima(arima_res, steps=len(test))
        arima_pred = pd.Series(arima_pred, index=test.index)
    buf = plot_series_inline(train, test, arima_pred, 'ARIMA Forecast')
    save_plot_buf(buf, os.path.join('plots', 'arima.png'))
    col_right.subheader("ARIMA Forecast")
    col_right.image(buf)
    col_right.markdown(f"Saved model: `models/arima_model.pkl`")

# Function to run Prophet
def run_prophet_and_show():
    with st.spinner("Training Prophet..."):
        prophet_model = train_prophet(train, save_path='models', name='prophet_model.pkl')
    with st.spinner("Forecasting Prophet..."):
        prophet_pred = forecast_prophet(prophet_model, periods=len(test))
        prophet_pred = pd.Series(prophet_pred.values, index=test.index)
    buf = plot_series_inline(train, test, prophet_pred, 'Prophet Forecast')
    save_plot_buf(buf, os.path.join('plots', 'prophet.png'))
    col_left.subheader("Prophet Forecast")
    col_left.image(buf)
    col_left.markdown(f"Saved model: `models/prophet_model.pkl`")

# Function to run LSTM
def run_lstm_and_show():
    with st.spinner("Training LSTM..."):
        # scale full series to create sequences
        scaler = MinMaxScaler()
        values = series.values.reshape(-1,1)
        scaler.fit(values)
        scaled = scaler.transform(values)
        split_idx = int(len(scaled)*0.8)
        train_s = scaled[:split_idx]
        test_s = scaled[split_idx:]
        lstm_model = train_lstm(train_s, seq_len=lstm_seq, epochs=int(lstm_epochs), batch_size=int(lstm_batch), save_path='models', name='lstm_model.h5')
    with st.spinner("Forecasting LSTM..."):
        lstm_pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
        lstm_pred = pd.Series(lstm_pred, index=test.index)
    buf = plot_series_inline(train, test, lstm_pred, 'LSTM Forecast')
    save_plot_buf(buf, os.path.join('plots', 'lstm.png'))
    col_right.subheader("LSTM Forecast")
    col_right.image(buf)
    col_right.markdown(f"Saved model: `models/lstm_model.h5`")

# Buttons / actions
if train_button or full_pipeline:
    to_run = []
    if full_pipeline:
        to_run = ["SARIMA", "ARIMA", "Prophet", "LSTM"]
    else:
        to_run = model_options

    st.info(f"Running: {to_run}")

    if "SARIMA" in to_run:
        try:
            run_sarima_and_show()
        except Exception as e:
            st.error(f"SARIMA failed: {e}")

    if "ARIMA" in to_run:
        try:
            run_arima_and_show()
        except Exception as e:
            st.error(f"ARIMA failed: {e}")

    if "Prophet" in to_run:
        try:
            run_prophet_and_show()
        except Exception as e:
            st.error(f"Prophet failed: {e}")

    if "LSTM" in to_run:
        try:
            run_lstm_and_show()
        except Exception as e:
            st.error(f"LSTM failed: {e}")

# Show existing plots & models
st.markdown("---")
st.subheader("Saved plots & models")
col_a, col_b = st.columns(2)

with col_a:
    st.write("Plots")
    for p in ['sarima.png', 'arima.png', 'prophet.png', 'lstm.png']:
        path = os.path.join('plots', p)
        if os.path.exists(path):
            st.image(path, caption=p, use_column_width='always')

with col_b:
    st.write("Saved models")
    for f in os.listdir('models'):
        st.write(f"- {f}")

st.markdown("### Logs / Troubleshooting")
st.write("If something fails, check the Streamlit Cloud logs (Manage app → Logs). Make sure `requirements.txt` includes all dependencies (statsmodels, prophet, tensorflow, scikit-learn, yfinance, etc.).")
