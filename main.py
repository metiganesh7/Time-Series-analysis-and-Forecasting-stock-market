import sys
import os
import io
import datetime as dt

# Ensure scripts folder import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Imports
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --------------------------
# SAFE PROPHET IMPORT
# --------------------------
try:
    from scripts.prophet_model import train_prophet, forecast_prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False
    PROPHET_ERROR = str(e)

# --------------------------
# COLORFUL BUTTON STYLING
# --------------------------
st.markdown("""
<style>
.stButton button {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    padding: 10px 18px;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}
.stDownloadButton button {
    background: linear-gradient(135deg, #ff9900, #ffcc00);
    color: black;
    padding: 8px 14px;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# METRIC FUNCTIONS
# --------------------------
def RMSE(a, p): return np.sqrt(mean_squared_error(a, p))
def MSE(a, p): return mean_squared_error(a, p)
def MAPE(a, p):
    a = np.array(a)
    p = np.array(p)
    a[a == 0] = 1e-8
    return np.mean(np.abs((a - p) / a)) * 100

# --------------------------
# HELPERS
# --------------------------
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower():
            return c
    return df.columns[0]

def detect_price_column(df):
    candidates = ["Close", "close", "Adj Close", "Price", "price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    return numeric[-1]

def plot_series_buf(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.legend()
    ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_combined_chart(train, test, preds):
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)

    for name, fc in preds.items():
        ax.plot(fc.index, fc.values, label=name, linewidth=2)

    ax.legend()
    ax.set_title("Combined Model Comparison")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_radar_chart(df):
    df = df[["RMSE", "MSE", "MAPE"]].astype(float)
    norm = (df.max() - df) / (df.max() - df.min() + 1e-8)

    labels = list(norm.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for idx in norm.index:
        values = norm.loc[idx].tolist() + [norm.loc[idx].tolist()[0]]
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf

# --------------------------
# UI
# --------------------------
st.title("📈 Stock Price Forecasting Dashboard")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
df[date_col] = pd.to_datetime(df[date_col])
df = df.set_index(date_col)

price_col = detect_price_column(df)
series = df[price_col]

st.subheader("📊 Data Preview")
st.dataframe(df.tail())

series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, 0.2)

# --------------------------
# MODEL SELECTOR
# --------------------------
model_opts = ["ARIMA", "SARIMA", "LSTM"]
if PROPHET_AVAILABLE:
    model_opts.append("Prophet")

models = st.sidebar.multiselect("Select Models", model_opts, default=model_opts)

if not PROPHET_AVAILABLE:
    st.sidebar.warning("⚠ Prophet disabled (Python 3.13 incompatible).")

# Hyperparameters
def parse_nums(txt, n):
    try: return tuple([int(x) for x in txt.split(",")][:n])
    except: return (1, 1, 1)[:n]

arima_order = parse_nums(st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0"), 3)
sarima_order = parse_nums(st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1"), 3)
seasonal_order = parse_nums(st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12"), 4)

lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_ep = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_bs = st.sidebar.number_input("Batch size", 1, 256, 32)

run = st.sidebar.button("🚀 Run Models")

# --------------------------
# RUNNING MODELS
# --------------------------
combined = {}
scores = {}

col1, col2 = st.columns(2)

if run:

    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            ar = train_arima(train.squeeze(), order=arima_order)
            fc = forecast_arima(ar, len(test))
            pred = pd.Series(fc, index=test.index)

            combined["ARIMA"] = pred
            scores["ARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}

            col1.subheader("ARIMA")
            col1.image(plot_series_buf(train, test, pred, "ARIMA Forecast"))

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            sa = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            fc = forecast_sarima(sa, len(test))
            pred = pd.Series(fc, index=test.index)

            combined["SARIMA"] = pred
            scores["SARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}

            col1.subheader("SARIMA")
            col1.image(plot_series_buf(train, test, pred, "SARIMA Forecast"))

    # PROPHET
    if "Prophet" in models and PROPHET_AVAILABLE:
        with st.spinner("Running Prophet..."):
            try:
                pr = train_prophet(train.squeeze())
                fc = forecast_prophet(pr, len(test)).reindex(test.index)
                pred = pd.Series(fc.values, index=test.index)

                combined["Prophet"] = pred
                scores["Prophet"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}

                col2.subheader("Prophet")
                col2.image(plot_series_buf(train, test, pred, "Prophet Forecast"))
            except Exception as e:
                st.error(f"Prophet failed: {e}")

    # LSTM
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            sc = MinMaxScaler()
            scaled = sc.fit_transform(series.values.reshape(-1, 1))
            split = int(len(scaled) * 0.8)
            train_scaled = scaled[:split]

            lstm = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_ep, batch_size=lstm_bs)
            fc = forecast_lstm(lstm, scaled, sc, seq_len=lstm_seq, steps=len(test))
            pred = pd.Series(fc, index=test.index)

            combined["LSTM"] = pred
            scores["LSTM"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}

            col2.subheader("LSTM")
            col2.image(plot_series_buf(train, test, pred, "LSTM Forecast"))

# --------------------------
# COMBINED CHART & METRICS
# --------------------------
if combined:
    st.subheader("📌 Combined Forecast Chart")
    buf = plot_combined_chart(train, test, combined)
    st.image(buf)

    st.download_button("Download Combined Chart", buf.getvalue(),
                       "combined_chart.png", "image/png")

if scores:
    st.subheader("📈 Model Performance Metrics")
    dfm = pd.DataFrame(scores).T.sort_values("RMSE")
    dfm["Rank"] = range(1, len(dfm) + 1)
    st.dataframe(dfm)

    st.success(f"🏆 Best Model: {dfm.index[0]}")

    radar = create_radar_chart(dfm)
    st.subheader("📊 Radar Chart")
    st.image(radar)

    st.download_button("Download Radar Chart", radar.getvalue(),
                       "radar_chart.png", "image/png")
