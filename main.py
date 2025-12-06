# ============================================================
# FULL STOCK FORECASTING APP + LIVE NSE CANDLESTICK CHART
# ============================================================

import sys
import os
import io

# Add /scripts folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import forecasting modules
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

# NSE API
from nsepython import nsefetch
import plotly.graph_objects as go

# =========================
# ERROR METRICS
# =========================

def RMSE(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

def MSE(y, yhat):
    return mean_squared_error(y, yhat)

def MAPE(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    y[y == 0] = 1e-9
    return np.mean(np.abs((y - yhat) / y)) * 100

# =========================
# COLUMN DETECTION
# =========================

def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close", "close", "Adj Close", "Price", "price"]:
        if c in df.columns:
            return c
    return df.select_dtypes(include="number").columns[-1]

# =========================
# PLOT HELPERS
# =========================

def plot_series(train, test, pred, title):
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


def plot_comparison(train, test, preds):
    fig, ax = plt.subplots(figsize=(12, 5))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    for name, p in preds.items():
        p.plot(ax=ax, label=name)
    ax.legend()
    ax.set_title("Model Comparison")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def radar_chart(df):
    df = df[["RMSE", "MSE", "MAPE"]]
    labels = df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    for model in df.index:
        vals = df.loc[model].values.astype(float)
        mn, mx = vals.min(), vals.max()
        norm = (mx - vals) / (mx - mn + 1e-9)
        norm = np.concatenate([norm, [norm[0]]])

        ax.plot(angles, norm, label=model)
        ax.fill(angles, norm, alpha=0.2)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Model Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# ============================================================
# UI TITLE
# ============================================================

st.title("📈 Advanced Stock Forecasting + Live NSE Candlestick Dashboard")

# ============================================================
# CSV Upload Section
# ============================================================

st.sidebar.header("Upload CSV File")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    date_col = detect_date_column(df)
    price_col = detect_price_column(df)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.tail())

    # Prepare series
    series = prepare_series(df, price_col, freq="D")
    train, test = train_test_split_series(series, 0.2)

    # --------------------------------------
    # Forecast Model Selection
    # --------------------------------------
    st.sidebar.header("Select Models")
    models = st.sidebar.multiselect(
        "Choose forecasting models",
        ["ARIMA","SARIMA","Prophet","LSTM"],
        default=["ARIMA","SARIMA","Prophet","LSTM"]
    )

    arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
    sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
    seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

    lstm_seq = st.sidebar.number_input("LSTM seq length", 10, 200, 60)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
    lstm_batch = st.sidebar.number_input("LSTM batch size", 1, 200, 32)

    run = st.sidebar.button("🚀 Run Forecasting Models")

    combined_preds = {}
    scores = {}

    col1, col2 = st.columns(2)

    if run:
        # ARIMA
        if "ARIMA" in models:
            m = train_arima(train.squeeze(), order=arima_order)
            pred = pd.Series(forecast_arima(m, len(test)), index=test.index)
            combined_preds["ARIMA"] = pred
            scores["ARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("ARIMA Forecast")
            col1.image(plot_series(train, test, pred, "ARIMA"))

        # SARIMA
        if "SARIMA" in models:
            m = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred = pd.Series(forecast_sarima(m, len(test)), index=test.index)
            combined_preds["SARIMA"] = pred
            scores["SARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("SARIMA Forecast")
            col1.image(plot_series(train, test, pred, "SARIMA"))

        # Prophet
        if "Prophet" in models:
            m = train_prophet(train.squeeze())
            pred_raw = forecast_prophet(m, len(test)).reindex(test.index)
            pred = pd.Series(pred_raw.values, index=test.index)
            combined_preds["Prophet"] = pred
            scores["Prophet"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("Prophet Forecast")
            col2.image(plot_series(train, test, pred, "Prophet"))

        # LSTM
        if "LSTM" in models:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled)*0.8)
            lstm_train = scaled[:split]

            lstm_model = train_lstm(lstm_train, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            pred_vals = forecast_lstm(lstm_model, scaled, scaler, lstm_seq, len(test))
            pred = pd.Series(pred_vals, index=test.index)

            combined_preds["LSTM"] = pred
            scores["LSTM"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("LSTM Forecast")
            col2.image(plot_series(train, test, pred, "LSTM"))

        # --- Combined Comparison ---
        st.subheader("📌 Combined Forecast Comparison")
        st.image(plot_comparison(train, test, combined_preds))

        st.subheader("📊 Model Performance Metrics")
        metrics = pd.DataFrame(scores).T
        metrics = metrics.sort_values("RMSE")
        metrics["Rank"] = range(1, len(metrics)+1)
        st.dataframe(metrics)

        st.success(f"🏆 Best Model: {metrics.index[0]}")

        st.subheader("📡 Radar Chart")
        st.image(radar_chart(metrics))


# ============================================================
# LIVE NSE CANDLESTICK CHART (WORKING)
# ============================================================

st.markdown("---")
st.header("📈 Live NSE Candlestick Chart (Official NSE Data)")

symbol_input = st.text_input(
    "Enter NSE Symbol (HDFCBANK, TCS, RELIANCE, INFY, WIPRO, etc)",
    "HDFCBANK"
)

if symbol_input:
    try:
        symbol = symbol_input.upper().replace(".NS", "")
        api_url = (
            f"https://www.nseindia.com/api/historical/cm/equity"
            f"?symbol={symbol}&series=[%22EQ%22]&from=06-12-2024&to=06-12-2025"
        )

        data = nsefetch(api_url)

        if "data" not in data or len(data["data"]) == 0:
            st.error("⚠ NSE returned no data. Check symbol (don’t use .NS).")
        else:
            df = pd.DataFrame(data["data"])
            df["date"] = pd.to_datetime(df["CH_TIMESTAMP"])
            df.set_index("date", inplace=True)

            fig = go.Figure(go.Candlestick(
                x=df.index,
                open=df["CH_OPENING_PRICE"],
                high=df["CH_TRADE_HIGH_PRICE"],
                low=df["CH_TRADE_LOW_PRICE"],
                close=df["CH_CLOSING_PRICE"],
                name=symbol
            ))

            fig.update_layout(
                title=f"{symbol} – NSE Candlestick Chart (1 Year)",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠ Error loading NSE data: {e}")
