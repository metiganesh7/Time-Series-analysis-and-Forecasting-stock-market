import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys, os

# =========================
# ✅ IMPORT MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# =========================
# ✅ SIGNAL + RISK SYSTEM
# =========================
def generate_signal(last_price, forecast_price):
    change_pct = ((forecast_price - last_price) / last_price) * 100
    if change_pct > 1:
        return "BUY", change_pct
    elif change_pct < -1:
        return "SELL", change_pct
    else:
        return "HOLD", change_pct

def calculate_target_stop(price, signal, target_pct=0.8, stop_pct=0.5):
    if signal == "BUY":
        target = price * (1 + target_pct/100)
        stop = price * (1 - stop_pct/100)
    elif signal == "SELL":
        target = price * (1 - target_pct/100)
        stop = price * (1 + stop_pct/100)
    else:
        target, stop = None, None
    return target, stop

def calculate_position_size(capital, entry_price, stop_price, risk_percent=1):
    risk_amount = capital * (risk_percent / 100)
    per_share_risk = abs(entry_price - stop_price)
    if per_share_risk == 0:
        return 0, 0, 0
    quantity = int(risk_amount / per_share_risk)
    position_value = quantity * entry_price
    return quantity, position_value, risk_amount

def backtest_strategy(series):
    capital = 100000
    position = 0
    equity = []

    for i in range(1, len(series)):
        signal = "BUY" if series.iloc[i] > series.iloc[i-1] else "SELL"

        if signal == "BUY" and capital > 0:
            position = capital / series.iloc[i]
            capital = 0
        elif signal == "SELL" and position > 0:
            capital = position * series.iloc[i]
            position = 0

        equity.append(capital + position * series.iloc[i])

    equity = pd.Series(equity, index=series.index[1:])
    total_return = (equity.iloc[-1] / 100000 - 1) * 100
    win_rate = (equity.pct_change().fillna(0) > 0).mean() * 100
    drawdown = (equity / equity.cummax() - 1).min() * 100
    return equity, total_return, win_rate, drawdown

# =========================
# ✅ UI
# =========================
st.set_page_config(page_title="Intraday Trading Pro", layout="wide")
st.title("⚡ Intraday Forecasting & Trading System")

# =========================
# ✅ UPLOAD CSV
# =========================
file = st.file_uploader("📂 Upload Intraday CSV (1m / 5m / 15m / 30m)", type=["csv"])

if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

# ✅ Detect datetime & price
date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
price_col = df.select_dtypes(include="number").columns[-1]

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

# =========================
# ✅ INTRADAY TIMEFRAME SELECTOR
# =========================
tf_map = {
    "1 Minute": "1T",
    "5 Minutes": "5T",
    "15 Minutes": "15T",
    "30 Minutes": "30T"
}

tf = st.selectbox("⏱ Select Intraday Timeframe", list(tf_map.keys()))

df = df.resample(tf_map[tf]).last().dropna()

series = df[price_col].astype(float)

st.line_chart(series.tail(200))

# =========================
# ✅ FORECAST SETTINGS
# =========================
horizon = st.selectbox("📅 Candles Ahead", [10, 20, 30, 50], index=1)

series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.8)

# =========================
# ✅ RUN SYSTEM
# =========================
if st.button("🚀 Run Intraday AI Trading System"):

    future_index = pd.date_range(series.index[-1], periods=horizon+1, freq=tf_map[tf])[1:]

    # ---------- ARIMA ----------
    arima = train_arima(train, order=(5,1,0))
    future = forecast_arima(arima, horizon)
    arima_pred = pd.Series(future, index=future_index)

    # =========================
    # ✅ COMBINED FORECAST
    # =========================
    st.subheader("📊 Intraday Forecast")
    fig, ax = plt.subplots(figsize=(12,5))
    series.tail(200).plot(ax=ax, label="Actual")
    arima_pred.plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # ✅ SIGNAL
    # =========================
    last_price = series.iloc[-1]
    future_price = arima_pred.iloc[-1]
    signal, strength = generate_signal(last_price, future_price)
    target, stop = calculate_target_stop(last_price, signal)

    st.subheader("📢 Intraday Trade Signal")
    st.metric("Signal", signal)
    st.metric("Expected Move %", f"{strength:.2f}%")

    if target:
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry", f"{last_price:.2f}")
        c2.metric("Target", f"{target:.2f}")
        c3.metric("Stop", f"{stop:.2f}")

        capital = st.number_input("Trading Capital", 1000, 10_000_000, 100000, step=1000)
        risk_percent = st.slider("Risk % per Trade", 0.5, 5.0, 1.0, step=0.5)

        qty, pos_value, risk_amt = calculate_position_size(
            capital, last_price, stop, risk_percent
        )

        st.markdown("### 📦 Position Sizing")
        c1, c2, c3 = st.columns(3)
        c1.metric("Quantity", qty)
        c2.metric("Position Value", f"{pos_value:,.0f}")
        c3.metric("Max Risk", f"{risk_amt:,.0f}")

    # =========================
    # ✅ BACKTESTING
    # =========================
    st.subheader("📊 Intraday Strategy Backtest")

    equity, total_return, win_rate, drawdown = backtest_strategy(series[-500:])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Return %", f"{total_return:.2f}%")
    c2.metric("Win Rate %", f"{win_rate:.2f}%")
    c3.metric("Max Drawdown %", f"{drawdown:.2f}%")

    st.line_chart(equity)
