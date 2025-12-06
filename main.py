# ============================================================
# FULL PREMIUM STOCK FORECASTING APP + LIVE NSE CHART
# ============================================================

import streamlit as st

# ============================================================
# PREMIUM UI STYLING (Glassmorphism + Neon Glow)
# ============================================================
st.markdown("""
<style>

html, body, .stApp {
    background: linear-gradient(135deg, #0a0f1f, #08111f, #050a0e);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: #d8e1f0 !important;
    font-family: 'Roboto', sans-serif;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Main Container */
.block-container {
    backdrop-filter: blur(14px) saturate(180%);
    background: rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2rem !important;
    box-shadow: 0 0 25px rgba(0, 200, 255, 0.15);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    backdrop-filter: blur(10px) saturate(150%);
    background: rgba(255, 255, 255, 0.05);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

h1, h2, h3 {
    color: #e8f1ff !important;
    text-shadow: 0px 0px 12px rgba(0,200,255,0.35);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #008cff, #00d4ff);
    border: none;
    color: white;
    padding: 0.65rem 1.3rem;
    border-radius: 10px;
    font-size: 1.1rem;
    transition: 0.25s ease;
    box-shadow: 0px 0px 12px rgba(0,200,255,0.5);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 16px rgba(0,220,255,0.85);
}

/* Inputs */
input, select, textarea {
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e6f0ff !important;
}

/* Animations */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
div[data-testid="stAppViewContainer"] {
    animation: fadeIn 0.8s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# IMPORTS
# ============================================================

import sys
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from nsepython import nsefetch

# Import forecasting models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
sys.path.append(SCRIPTS_DIR)

from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def RMSE(y, p): return np.sqrt(mean_squared_error(y, p))
def MSE(y, p): return mean_squared_error(y, p)
def MAPE(y, p):
    y = np.array(y); p = np.array(p)
    y[y == 0] = 1e-9
    return np.mean(np.abs((y - p) / y)) * 100

# ============================================================
# HELPERS
# ============================================================

def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower(): return c
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close","close","Adj Close","Price"]:
        if c in df.columns: return c
    return df.select_dtypes(include="number").columns[-1]

def buf_plot(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax); test.plot(ax=ax); pred.plot(ax=ax)
    ax.set_title(title); ax.legend()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
    plt.close(fig); return buf

# ============================================================
# TITLE
# ============================================================

st.title("📈 Premium Stock Forecasting + Live NSE Dashboard")

# ============================================================
# CSV UPLOAD
# ============================================================

st.sidebar.header("Upload CSV File")
file = st.sidebar.file_uploader("Upload a stock CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    date_col = detect_date_column(df)
    price_col = detect_price_column(df)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    st.subheader("📊 Uploaded Stock Preview")
    st.dataframe(df.tail())

    # Prepare series
    series = prepare_series(df, price_col, freq="D")
    train, test = train_test_split_series(series, 0.2)

    # ============================================================
    # MODEL SELECTION
    # ============================================================

    st.sidebar.header("Forecast Models")
    models = st.sidebar.multiselect(
        "Choose models",
        ["ARIMA","SARIMA","Prophet","LSTM"],
        default=["ARIMA","SARIMA","Prophet","LSTM"]
    )

    arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
    sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
    seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

    lstm_seq = st.sidebar.number_input("LSTM seq_len", 10, 200, 60)
    lstm_epochs = st.sidebar.number_input("Epochs", 1, 50, 5)
    lstm_batch = st.sidebar.number_input("Batch size", 1, 256, 32)

    run = st.sidebar.button("🚀 Run Forecasting")

    forecasts = {}
    metrics = {}

    col1, col2 = st.columns(2)

    # ============================================================
    # RUN MODELS
    # ============================================================

    if run:

        # ----- ARIMA -----
        if "ARIMA" in models:
            m = train_arima(train.squeeze(), order=arima_order)
            pred = pd.Series(forecast_arima(m, len(test)), index=test.index)

            forecasts["ARIMA"] = pred
            metrics["ARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }

            col1.subheader("ARIMA Forecast")
            col1.image(buf_plot(train, test, pred, "ARIMA"))

        # ----- SARIMA -----
        if "SARIMA" in models:
            m = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred = pd.Series(forecast_sarima(m, len(test)), index=test.index)

            forecasts["SARIMA"] = pred
            metrics["SARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }

            col1.subheader("SARIMA Forecast")
            col1.image(buf_plot(train, test, pred, "SARIMA"))

        # ----- Prophet -----
        if "Prophet" in models:
            m = train_prophet(train.squeeze())
            pred_raw = forecast_prophet(m, len(test)).reindex(test.index)

            pred = pd.Series(pred_raw.values, index=test.index)
            forecasts["Prophet"] = pred
            metrics["Prophet"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }

            col2.subheader("Prophet Forecast")
            col2.image(buf_plot(train, test, pred, "Prophet"))

        # ----- LSTM -----
        if "LSTM" in models:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            lstm_train = scaled[:int(len(scaled)*0.8)]

            lstm_model = train_lstm(lstm_train, seq_len=lstm_seq,
                epochs=lstm_epochs, batch_size=lstm_batch)

            pred_vals = forecast_lstm(lstm_model, scaled, scaler, lstm_seq, len(test))
            pred = pd.Series(pred_vals, index=test.index)

            forecasts["LSTM"] = pred
            metrics["LSTM"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred)
            }

            col2.subheader("LSTM Forecast")
            col2.image(buf_plot(train, test, pred, "LSTM"))

        # ============================================================
        # METRICS & RANKING
        # ============================================================

        st.markdown("---")
        st.subheader("📊 Model Performance")

        dfm = pd.DataFrame(metrics).T
        dfm = dfm.sort_values("RMSE")
        dfm["Rank"] = range(1, len(dfm)+1)

        st.dataframe(dfm)
        st.success(f"🏆 Best Model: **{dfm.index[0]}**")

# ============================================================
# LIVE NSE CANDLESTICK CHART
# ============================================================

st.markdown("---")
st.header("📈 Live NSE Candlestick Chart (NSE Official API)")

symbol = st.text_input("Enter NSE symbol (HDFCBANK, RELIANCE, TCS)", "HDFCBANK")

if symbol:
    try:
        symbol_clean = symbol.upper().replace(".NS", "")
        url = (
            f"https://www.nseindia.com/api/historical/cm/equity?"
            f"symbol={symbol_clean}&series=[%22EQ%22]&from=06-12-2024&to=06-12-2025"
        )

        data = nsefetch(url)

        if "data" not in data or len(data["data"]) == 0:
            st.error("⚠ No data from NSE. Try symbol WITHOUT .NS")
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
                name=symbol_clean
            ))

            fig.update_layout(
                title=f"{symbol_clean} – NSE Candlestick Chart (1 Year)",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠ NSE API Error: {e}")
