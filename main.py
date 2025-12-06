import sys
import os
import io
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model scripts (must exist in scripts/)
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
from sklearn.metrics import mean_squared_error

# --------------------------
# THEME + RELIABLE ANIMATED BACKGROUND (CSS ONLY)
# --------------------------
st.markdown("""
<style>
/* Base */
.stApp {
    position: relative;
    overflow: hidden;
    background: radial-gradient(circle at 10% 10%, #061226 0%, #02040a 35%, #02030a 100%);
    color: #e6eef8;
    font-family: 'Poppins', sans-serif;
}

/* Particle-like layered animated blobs (CSS-only, reliable) */
.stApp::before{
  content: "";
  position: fixed;
  top: -10%;
  left: -10%;
  width: 220%;
  height: 220%;
  z-index: -3;
  background-image:
    radial-gradient(circle, rgba(0,230,255,0.10) 1px, transparent 1px),
    radial-gradient(circle, rgba(0,180,255,0.07) 1px, transparent 1px),
    radial-gradient(circle, rgba(180,240,255,0.03) 1px, transparent 1px);
  background-size: 120px 120px, 80px 80px, 50px 50px;
  animation: particleMove 30s linear infinite;
  opacity: 0.9;
  pointer-events: none;
}

.stApp::after{
  content: "";
  position: fixed;
  top: -10%;
  left: -10%;
  width: 220%;
  height: 220%;
  z-index: -2;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,0.01) 0%, transparent 40%),
    radial-gradient(circle at 30% 20%, rgba(0,255,200,0.02), transparent 10%);
  background-size: 400px 400px;
  animation: drift 60s linear infinite;
  opacity: 0.7;
  pointer-events: none;
}

@keyframes particleMove {
  0%   { transform: translate(0px, 0px) rotate(0deg) scale(1); }
  25%  { transform: translate(-40px, -20px) rotate(0.01turn) scale(1.01); }
  50%  { transform: translate(-80px, -40px) rotate(0.02turn) scale(1); }
  75%  { transform: translate(-40px, -20px) rotate(0.01turn) scale(0.99); }
  100% { transform: translate(0px, 0px) rotate(0turn) scale(1); }
}

@keyframes drift {
  0%   { transform: translate(0px, 0px) rotate(0deg); }
  50%  { transform: translate(-80px, -40px) rotate(0.005turn); }
  100% { transform: translate(0px, 0px) rotate(0deg); }
}

/* faint moving grid for depth */
.stApp .grid {
  position: fixed;
  inset: 0;
  z-index: -1;
  background-image:
    linear-gradient(rgba(0,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,255,0.02) 1px, transparent 1px);
  background-size: 200px 200px;
  animation: gridScroll 40s linear infinite;
  pointer-events: none;
  opacity: 0.45;
}

@keyframes gridScroll {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-60px); }
  100% { transform: translateY(0px); }
}

/* readable content card */
.block-container {
  backdrop-filter: blur(6px) saturate(1.1);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.03);
}

/* Buttons and download buttons */
.stButton button {
  background: linear-gradient(135deg, #6a11cb, #00d4ff);
  color: white;
  padding: 10px 16px;
  border-radius: 10px;
  border: none;
  font-weight: 600;
}
.stDownloadButton button {
  background: linear-gradient(135deg, #ffbe0b, #fb5607);
  color: black;
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 700;
}

/* small screens: reduce animations for perf */
@media (max-width: 600px) {
  .stApp::before, .stApp::after { display: none; }
}
</style>

<div class="grid"></div>
""", unsafe_allow_html=True)

# --------------------------
# METRIC FUNCTIONS
# --------------------------
def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def MAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual == 0
    if mask.any():
        actual = actual.copy()
        actual[mask] = 1e-8
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# --------------------------
# HELPERS
# --------------------------
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]

def detect_price_column(df):
    candidates = ["Close", "Adj Close", "close", "price", "Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("No numeric columns found")
    return numeric[-1]

def plot_series_buf(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_combined_chart(train, test, combined_predictions):
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)
    for name, fc in combined_predictions.items():
        try:
            fc.plot(ax=ax, label=name, linewidth=2)
        except Exception:
            ax.plot(fc.index, fc.values, label=name, linewidth=2)
    ax.set_title("Model Comparison Chart")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_radar_chart(metrics_df):
    df = metrics_df[["RMSE","MSE","MAPE"]].copy()
    norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].values.astype(float)
        mn, mx = vals.min(), vals.max()
        if mx - mn == 0:
            norm[col] = 1.0
        else:
            norm[col] = (mx - vals) / (mx - mn)
    labels = list(norm.columns)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    for idx in norm.index:
        values = norm.loc[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx, linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    ax.set_title("Model Radar Chart (Higher = Better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# UI layout
# --------------------------
st.title("📈 Stock Price Forecasting Dashboard")

st.sidebar.header("Upload CSV")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if not file:
    st.warning("Upload a CSV to continue.")
    st.stop()

# read uploaded dataset
try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# detect date and price columns
try:
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
except Exception as e:
    st.error(f"Date column error: {e}")
    st.stop()

try:
    price_col = detect_price_column(df)
except Exception as e:
    st.error(f"Price column error: {e}")
    st.stop()

series = df[price_col]

with st.expander("📊 Data Preview", True):
    st.dataframe(df.tail())
    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

# prepare series and train/test split
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)

# sidebar controls
st.sidebar.header("Models")
models = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA","SARIMA","Prophet","LSTM"],
    default=["ARIMA","SARIMA","Prophet","LSTM"]
)

# parse hyperparameters safely
def parse_tuple_input(s, length=3, default=(1,1,1)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) >= length:
            return tuple(parts[:length])
        else:
            # pad with defaults
            return tuple((parts + list(default))[0:length])
    except Exception:
        return default

arima_order = parse_tuple_input(st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0"), length=3, default=(5,1,0))
sarima_order = parse_tuple_input(st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1"), length=3, default=(1,1,1))
# seasonal expects 4
try:
    seasonal_text = st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12")
    seasonal_order = tuple([int(x.strip()) for x in seasonal_text.split(",")][:4])
    if len(seasonal_order) < 4:
        seasonal_order = (1,1,1,12)
except Exception:
    seasonal_order = (1,1,1,12)

lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)

run = st.sidebar.button("Run Models 🚀")

combined_predictions = {}
model_scores = {}

col1, col2 = st.columns(2)

# --------------------------
# Run models
# --------------------------
if run:
    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            try:
                arima_res = train_arima(train.squeeze(), order=arima_order)
                arima_vals = forecast_arima(arima_res, steps=len(test))
                pred = pd.Series(arima_vals, index=test.index)
                combined_predictions["ARIMA"] = pred
                model_scores["ARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
                buf = plot_series_buf(train, test, pred, "ARIMA Forecast")
                col1.subheader("ARIMA Forecast")
                col1.image(buf)
            except Exception as e:
                st.error(f"ARIMA error: {e}")

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            try:
                sarima_res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
                sarima_vals = forecast_sarima(sarima_res, steps=len(test))
                pred = pd.Series(sarima_vals, index=test.index)
                combined_predictions["SARIMA"] = pred
                model_scores["SARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
                buf = plot_series_buf(train, test, pred, "SARIMA Forecast")
                col1.subheader("SARIMA Forecast")
                col1.image(buf)
            except Exception as e:
                st.error(f"SARIMA error: {e}")

    # Prophet
    if "Prophet" in models:
        with st.spinner("Running Prophet..."):
            try:
                prophet_input = train.squeeze()
                prophet_model = train_prophet(prophet_input)
                prophet_pred_series = forecast_prophet(prophet_model, periods=len(test))
                prophet_pred_series = prophet_pred_series.reindex(test.index)
                pred = pd.Series(prophet_pred_series.values, index=test.index)
                combined_predictions["Prophet"] = pred
                model_scores["Prophet"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
                buf = plot_series_buf(train, test, pred, "Prophet Forecast")
                col2.subheader("Prophet Forecast")
                col2.image(buf)
            except Exception as e:
                st.error(f"Prophet error: {e}")

    # LSTM
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            try:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(series.values.reshape(-1,1))
                split = int(len(scaled)*0.8)
                train_scaled = scaled[:split]
                lstm_model = train_lstm(train_scaled, seq_len=int(lstm_seq), epochs=int(lstm_epochs), batch_size=int(lstm_batch))
                lstm_vals = forecast_lstm(lstm_model, scaled, scaler, seq_len=int(lstm_seq), steps=len(test))
                pred = pd.Series(lstm_vals, index=test.index)
                combined_predictions["LSTM"] = pred
                model_scores["LSTM"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
                buf = plot_series_buf(train, test, pred, "LSTM Forecast")
                col2.subheader("LSTM Forecast")
                col2.image(buf)
            except Exception as e:
                st.error(f"LSTM error: {e}")

    # --------------------------
    # Combined chart + download
    # --------------------------
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Forecast Chart")
        combined_buf = plot_combined_chart(train, test, combined_predictions)
        st.image(combined_buf)
        st.download_button("Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_chart.png", mime="image/png")

        # combined CSV (Actual + forecasts)
        combined_df = pd.DataFrame(index=test.index)
        for name, s in combined_predictions.items():
            combined_df[name] = s.values
        combined_df["Actual"] = test.values
        combined_csv = combined_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
        st.download_button("Download Combined Forecasts CSV", combined_csv, file_name="combined_forecasts.csv", mime="text/csv")

    # --------------------------
    # Metrics + ranking + radar + download
    # --------------------------
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Performance Metrics")
        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values("RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df)+1)
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues").format({
            "RMSE": "{:.4f}",
            "MSE": "{:.4f}",
            "MAPE": "{:.2f}%"
        }))
        best = metrics_df.index[0]
        st.success(f"🏆 Best Model: **{best}**")
        metrics_csv = metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics CSV", metrics_csv, file_name="metrics.csv", mime="text/csv")

        # radar chart
        radar_buf = create_radar_chart(metrics_df)
        st.subheader("📡 Radar Chart (Inverted Metrics)")
        st.image(radar_buf)
        st.download_button("Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")
