import sys
import os
import io
import datetime as dt

# Add scripts folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model scripts
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------
# 🔥 SMART SYMBOL DETECTION + CORRECTION
# ---------------------------------------------------------

NSE_MAP = {
    "HDFC": "HDFCBANK",
    "HDFCBANK": "HDFCBANK",
    "RELIANCE": "RELIANCE",
    "TCS": "TCS",
    "INFY": "INFY",
    "WIPRO": "WIPRO",
    "SBIN": "SBIN",
    "ICICIBANK": "ICICIBANK",
    "ASIANPAINT": "ASIANPAINT",
    "MARUTI": "MARUTI",
    "KOTAKBANK": "KOTAKBANK",
    "AXISBANK": "AXISBANK",
    "HCLTECH": "HCLTECH",
    "ULTRACEMCO": "ULTRACEMCO",
    "ADANIPORTS": "ADANIPORTS",
    "ADANIPOWER": "ADANIPOWER",
    "ADANIENT": "ADANIENT",
    "MRF": "MRF",
}

def resolve_symbol(file_name: str) -> str:
    """Convert uploaded CSV → best NSE TradingView symbol."""
    base = os.path.splitext(file_name)[0].upper()
    base = base.replace(" ", "").replace("-", "").replace("_", "")

    # Correct special cases (HDFC → HDFCBANK)
    if base in NSE_MAP:
        return f"NSE:{NSE_MAP[base]}"

    # Remove `.NS`
    if base.endswith(".NS"):
        base = base.replace(".NS", "")

    return f"NSE:{base}"


def normalize_symbol(sym: str) -> str:
    """
    Normalize user input into proper TradingView symbol format.
    Examples:
      HDFC.NS → NSE:HDFC → corrected to NSE:HDFCBANK
      RELIANCE → NSE:RELIANCE
      NSE:TCS.NS → NSE:TCS
    """
    if not sym:
        return ""

    sym = sym.upper().strip()

    # User enters HDFC.NS → NSE:HDFC
    if sym.endswith(".NS"):
        sym = sym.replace(".NS", "")
        sym = f"NSE:{sym}"

    # User enters NSE:HDFC.NS → NSE:HDFC
    if sym.startswith("NSE:") and sym.endswith(".NS"):
        sym = sym.replace(".NS", "")

    # Plain stock name → NSE:NAME
    if sym.isalpha() and ":" not in sym:
        sym = f"NSE:{sym}"

    # FINAL CORRECTION: HDFC → HDFCBANK
    base = sym.replace("NSE:", "")
    if base in NSE_MAP:
        return f"NSE:{NSE_MAP[base]}"

    return sym


# ---------------------------------------------------------
# METRIC FUNCTIONS
# ---------------------------------------------------------

def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def MAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    actual = np.where(actual == 0, 1e-8, actual)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# ---------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------

def plot_series_buf(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10, 4))
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
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)

    for name, series in combined_predictions.items():
        series.plot(ax=ax, label=name, linewidth=2)

    ax.set_title("Model Comparison Forecast")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


def create_radar_chart(metrics_df):
    df = metrics_df[["RMSE", "MSE", "MAPE"]].astype(float)
    norm = (df.max() - df) / (df.max() - df.min()).replace(0, 1)

    labels = df.columns
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for idx in norm.index:
        values = norm.loc[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Model Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------------------------------------
# TRADINGVIEW DASHBOARDS
# ---------------------------------------------------------

def tradingview_chart(symbol):
    widget = f"""
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>

      <script type="text/javascript" 
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js">
      {{
        "autosize": true,
        "symbol": "{symbol}",
        "interval": "D",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "allow_symbol_change": true,
        "calendar": false
      }}
      </script>
    </div>
    """
    components.html(widget, height=600)


def tradingview_screener():
    widget = """
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>

      <script type="text/javascript"
      src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js">
      {
        "width": "100%",
        "height": 620,
        "defaultColumn": "overview",
        "defaultScreen": "general",
        "market": "india",
        "colorTheme": "dark",
        "locale": "en"
      }
      </script>
    </div>
    """
    components.html(widget, height=640)


# ---------------------------------------------------------
# STREAMLIT APP UI
# ---------------------------------------------------------

st.set_page_config(page_title="AI Stock Forecasting Dashboard", layout="wide")
st.title("📈 Premium Stock Forecasting Dashboard")


# ------------------ FILE UPLOAD ------------------

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.warning("Upload a CSV file.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

# Detect date column
date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

# Detect price column
price_candidates = ["Close", "Adj Close", "Price"]
price_col = next((c for c in price_candidates if c in df.columns),
                 df.select_dtypes(include="number").columns[-1])

series = df[price_col]

with st.expander("📊 Data Overview", True):
    st.dataframe(df.tail())


# ---------------- SYMBOL HANDLING ----------------

auto_symbol = resolve_symbol(uploaded.name)

manual_input = st.sidebar.text_input("TradingView Symbol Override", auto_symbol)
tv_symbol = normalize_symbol(manual_input)

st.subheader("📉 Live TradingView Chart")
tradingview_chart(tv_symbol)


st.subheader("📋 TradingView Stock Screener")
tradingview_screener()


# ---------------------------------------------------------
# MODEL TRAINING SECTION
# ---------------------------------------------------------

series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)

st.sidebar.subheader("Models")
models = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
)

arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

lstm_seq = st.sidebar.number_input("LSTM Sequence Length", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM Batch Size", 1, 256, 32)

run = st.sidebar.button("🚀 Run Models")

combined_predictions = {}
model_scores = {}

if run:

    col1, col2 = st.columns(2)

    # ---------------- ARIMA ----------------
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            model = train_arima(train.squeeze(), order=arima_order)
            pred = pd.Series(forecast_arima(model, len(test)), index=test.index)
            combined_predictions["ARIMA"] = pred
            model_scores["ARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("ARIMA Forecast")
            col1.image(plot_series_buf(train, test, pred, "ARIMA Forecast"))

    # ---------------- SARIMA ----------------
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            model = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred = pd.Series(forecast_sarima(model, len(test)), index=test.index)
            combined_predictions["SARIMA"] = pred
            model_scores["SARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("SARIMA Forecast")
            col1.image(plot_series_buf(train, test, pred, "SARIMA Forecast"))

    # ---------------- Prophet ----------------
    if "Prophet" in models:
        with st.spinner("Running Prophet..."):
            model = train_prophet(train.squeeze())
            pred_vals = forecast_prophet(model, len(test)).reindex(test.index)
            pred = pd.Series(pred_vals.values, index=test.index)
            combined_predictions["Prophet"] = pred
            model_scores["Prophet"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("Prophet Forecast")
            col2.image(plot_series_buf(train, test, pred, "Prophet Forecast"))

    # ---------------- LSTM ----------------
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled) * 0.8)
            train_scaled = scaled[:split]

            lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            preds = forecast_lstm(lstm_model, scaled, scaler, lstm_seq, len(test))
            pred = pd.Series(preds, index=test.index)
            combined_predictions["LSTM"] = pred
            model_scores["LSTM"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("LSTM Forecast")
            col2.image(plot_series_buf(train, test, pred, "LSTM Forecast"))

    # ---------------- Combined Chart ----------------
    st.subheader("📊 Combined Forecast Chart")
    combined_buf = plot_combined_chart(train, test, combined_predictions)
    st.image(combined_buf)
    st.download_button("Download Combined Chart", combined_buf.getvalue(), "combined_chart.png")

    # ---------------- Metrics ----------------
    st.subheader("📈 Model Performance Metrics")

    metrics_df = pd.DataFrame(model_scores).T
    metrics_df["Rank"] = metrics_df["RMSE"].rank().astype(int)

    st.dataframe(metrics_df.style.background_gradient(cmap="Blues"))

    best_model = metrics_df.sort_values("RMSE").index[0]
    st.success(f"🏆 Best Model: {best_model}")

    st.download_button("Download Metrics CSV", metrics_df.to_csv().encode(), "metrics.csv")

    st.subheader("📡 Radar Chart")
    radar_buf = create_radar_chart(metrics_df)
    st.image(radar_buf)
    st.download_button("Download Radar Chart", radar_buf.getvalue(), "radar_chart.png")
