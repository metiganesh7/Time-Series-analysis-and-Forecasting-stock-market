import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# =========================
# ✅ IMPORT MODELS
# =========================
import sys, os

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
# ✅ BUY / SELL SIGNAL
# =========================
def generate_signal(last_price, forecast_price):
    change_pct = ((forecast_price - last_price) / last_price) * 100
    if change_pct > 2:
        return "BUY", change_pct
    elif change_pct < -2:
        return "SELL", change_pct
    else:
        return "HOLD", change_pct

# =========================
# ✅ TARGET & STOP LOSS
# =========================
def calculate_target_stop(price, signal, target_pct=5, stop_pct=3):
    if signal == "BUY":
        target = price * (1 + target_pct/100)
        stop = price * (1 - stop_pct/100)
    elif signal == "SELL":
        target = price * (1 - target_pct/100)
        stop = price * (1 + stop_pct/100)
    else:
        target, stop = None, None
    return target, stop

# =========================
# ✅ BACKTESTING LOGIC
# =========================
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
    returns = equity.pct_change().fillna(0)

    total_return = (equity.iloc[-1] / 100000 - 1) * 100
    win_rate = (returns > 0).mean() * 100
    drawdown = (equity / equity.cummax() - 1).min() * 100

    return equity, total_return, win_rate, drawdown

# =========================
# ✅ UI SETUP
# =========================
st.set_page_config(page_title="Stock Forecasting Pro", layout="wide")
st.title("📊 Stock Forecasting + Trading Strategy Platform")

# =========================
# ✅ HELPERS
# =========================
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close", "close", "Adj Close", "Price"]:
        if c in df.columns:
            return c
    nums = df.select_dtypes(include="number").columns
    return nums[-1] if len(nums)>0 else None

def plot_series(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# =========================
# ✅ UPLOAD CSV
# =========================
file = st.file_uploader("📂 Upload Stock CSV", type=["csv"])

if not file:
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
price_col = detect_price_column(df)

df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

series = df[price_col].astype(float)
series = prepare_series(df, price_col)

train, test = train_test_split_series(series, 0.2)

# =========================
# ✅ FORECAST HORIZON
# =========================
horizon = st.selectbox("📅 Forecast Days", [7,15,30,60], index=2)

if st.button("🚀 Run Forecast + Trading System"):

    preds = {}

    # =========================
    # ✅ ARIMA FORECAST
    # =========================
    arima = train_arima(train, order=(5,1,0))
    future = forecast_arima(arima, horizon)

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    preds["ARIMA"] = pd.Series(future, index=future_index)

    st.image(plot_series(train, test, preds["ARIMA"], "ARIMA Forecast"))

    # =========================
    # ✅ BUY / SELL SIGNAL
    # =========================
    last_price = series.iloc[-1]
    future_price = preds["ARIMA"].iloc[-1]
    signal, strength = generate_signal(last_price, future_price)

    st.subheader("📢 Trading Signal")
    st.metric("Expected Change %", f"{strength:.2f}%")
    st.success(f"Signal → {signal}")

    # =========================
    # ✅ TARGET & STOP LOSS
    # =========================
    target, stop = calculate_target_stop(last_price, signal)

    if target:
        st.subheader("🎯 Target & 🛑 Stop Loss")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"{last_price:.2f}")
        col2.metric("Target", f"{target:.2f}")
        col3.metric("Stop Loss", f"{stop:.2f}")

        rr = abs((target-last_price)/(last_price-stop)) if signal=="BUY" else 0
        st.metric("Risk-Reward Ratio", f"{rr:.2f}")

    # =========================
    # ✅ BACKTESTING
    # =========================
    st.subheader("📊 Trading Strategy Backtesting")

    equity, total_return, win_rate, drawdown = backtest_strategy(test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return %", f"{total_return:.2f}%")
    col2.metric("Win Rate %", f"{win_rate:.2f}%")
    col3.metric("Max Drawdown %", f"{drawdown:.2f}%")

    st.line_chart(equity)
