# ======================================================
# FULL APP CODE WITH NSE CANDLESTICK + FORECAST MODELS
# ======================================================

import sys
import os
import io
import datetime as dt

# ---------- Import model scripts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# ---------- General Libraries ----------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ---------- NSE Candlestick dependencies ----------
from nsepython import *
import plotly.graph_objects as go

# ======================================================
# METRIC FUNCTIONS
# ======================================================

def RMSE(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

def MSE(y, yhat):
    return mean_squared_error(y, yhat)

def MAPE(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    y[y == 0] = 1e-8
    return np.mean(np.abs((y - yhat) / y)) * 100


# ======================================================
# AUTO COLUMN HELPERS
# ======================================================

def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close", "close", "Adj Close", "Price", "price"]:
        if c in df.columns:
            return c
    # fallback: numeric last column
    return df.select_dtypes(include="number").columns[-1]


# ======================================================
# PLOT HELPERS
# ======================================================

def plot_series_buf(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
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


def plot_comparison(train, test, preds):
    fig, ax = plt.subplots(figsize=(12,5))
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
    df = df[["RMSE","MSE","MAPE"]]
    labels = df.columns.tolist()

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    for model in df.index:
        values = df.loc[model].values.astype(float)
        maxv = values.max()
        minv = values.min()
        norm = (maxv - values) / (maxv - minv + 1e-9)
        norm = np.concatenate([norm, [norm[0]]])
        ax.plot(angles, norm, label=model)
        ax.fill(angles, norm, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Model Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# ======================================================
# UI HEADER
# ======================================================

st.title("📈 Stock Forecasting + Live NSE Candlestick Dashboard")


# ======================================================
# CSV FILE UPLOAD
# ======================================================
st.sidebar.header("Upload Stock CSV")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if not file:
    st.info("Upload a CSV to begin forecasting.")
else:
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

    # ======================================================
    # SIDEBAR MODELS
    # ======================================================
    st.sidebar.header("Models")
    models = st.sidebar.multiselect(
        "Select Forecast Models",
        ["ARIMA","SARIMA","Prophet","LSTM"],
        default=["ARIMA","SARIMA","Prophet","LSTM"]
    )

    arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
    sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
    seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

    lstm_seq = st.sidebar.number_input("LSTM sequence length", 10, 200, 60)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
    lstm_batch = st.sidebar.number_input("LSTM batch size", 1, 256, 32)

    run = st.sidebar.button("🚀 Run Models")

    combined_predictions = {}
    model_scores = {}

    col1, col2 = st.columns(2)

    # ======================================================
    # RUN FORECAST MODELS
    # ======================================================
    if run:

        # ARIMA
        if "ARIMA" in models:
            res = train_arima(train.squeeze(), order=arima_order)
            pred = pd.Series(forecast_arima(res, len(test)), index=test.index)
            combined_predictions["ARIMA"] = pred
            model_scores["ARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("ARIMA Forecast")
            col1.image(plot_series_buf(train, test, pred, "ARIMA Forecast"))

        # SARIMA
        if "SARIMA" in models:
            res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred = pd.Series(forecast_sarima(res, len(test)), index=test.index)
            combined_predictions["SARIMA"] = pred
            model_scores["SARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col1.subheader("SARIMA Forecast")
            col1.image(plot_series_buf(train, test, pred, "SARIMA Forecast"))

        # Prophet
        if "Prophet" in models:
            model = train_prophet(train.squeeze())
            pred_vals = forecast_prophet(model, len(test))
            pred_vals = pred_vals.reindex(test.index)
            pred = pd.Series(pred_vals.values, index=test.index)
            combined_predictions["Prophet"] = pred
            model_scores["Prophet"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("Prophet Forecast")
            col2.image(plot_series_buf(train, test, pred, "Prophet Forecast"))

        # LSTM
        if "LSTM" in models:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled)*0.8)
            lstm_train = scaled[:split]
            model_lstm = train_lstm(lstm_train, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            pred_vals = forecast_lstm(model_lstm, scaled, scaler, lstm_seq, len(test))
            pred = pd.Series(pred_vals, index=test.index)
            combined_predictions["LSTM"] = pred
            model_scores["LSTM"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }
            col2.subheader("LSTM Forecast")
            col2.image(plot_series_buf(train, test, pred, "LSTM Forecast"))


        # ======================================================
        # COMBINED RESULTS
        # ======================================================
        st.subheader("📌 Combined Model Comparison")
        st.image(plot_comparison(train, test, combined_predictions))

        st.subheader("📊 Performance Metrics")
        dfm = pd.DataFrame(model_scores).T
        dfm = dfm.sort_values("RMSE")
        dfm["Rank"] = range(1, len(dfm)+1)
        st.dataframe(dfm)

        st.success(f"🏆 Best Model: {dfm.index[0]}")

        st.subheader("📡 Radar Chart")
        st.image(radar_chart(dfm))


# ======================================================
# LIVE NSE CANDLESTICK CHART (WORKS 100% ON STREAMLIT CLOUD)
# ======================================================

st.markdown("---")
st.header("📈 Live NSE Candlestick Chart (Official NSE Data)")

symbol = st.text_input(
    "Enter NSE Symbol (RELIANCE, TCS, HDFCBANK, INFY, WIPRO, ICICIBANK):",
    "HDFCBANK"
)

if symbol:
    try:
        df = nse_stock_history(symbol.upper(), "1Y")

        if df.empty:
            st.error("⚠ No data returned by NSE. Try another symbol.")
        else:
            df['date'] = pd.to_datetime(df['CH_TIMESTAMP'])
            df.set_index('date', inplace=True)

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['CH_OPENING_PRICE'],
                high=df['CH_TRADE_HIGH_PRICE'],
                low=df['CH_TRADE_LOW_PRICE'],
                close=df['CH_CLOSING_PRICE'],
                name=symbol
            ))

            fig.update_layout(
                title=f"{symbol} – 1 Year Candlestick Chart (NSE India)",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠ Error loading NSE data: {e}")
