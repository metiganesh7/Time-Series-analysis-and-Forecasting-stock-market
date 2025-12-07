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
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Stock Forecasting Pro", layout="wide")

# =========================
# ✅ PREMIUM UI THEME
# =========================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #020617, #02030f);
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #02030f);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.block-container {
    padding: 2rem 2.5rem;
}
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 40px rgba(56,189,248,0.12);
    margin-bottom: 20px;
}
.model-card {
    background: linear-gradient(135deg, #0f172a, #020617);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
}
.model-title {
    font-size: 20px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}
h1, h2, h3 {
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
.stButton button {
    background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
    color: white !important;
    font-weight: bold;
    border-radius: 14px;
    padding: 10px 24px;
    border: none;
    box-shadow: 0 0 15px rgba(124,58,237,0.4);
}
img {
    border-radius: 16px;
    box-shadow: 0 0 40px rgba(56,189,248,0.18);
}
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ HEADER
# =========================
st.markdown("""
<div class='glass-card'>
    <h1>📊 AI Stock Forecasting & Trading Dashboard</h1>
    <p style='color:#9ca3af'>
        Forecasting • Signal Generation • Risk Management • Backtesting • Position Sizing
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# ✅ HELPER FUNCTIONS
# =========================
def generate_signal(last_price, forecast_price):
    change_pct = ((forecast_price - last_price) / last_price) * 100
    if change_pct > 1:
        return "BUY", change_pct
    elif change_pct < -1:
        return "SELL", change_pct
    else:
        return "HOLD", change_pct

def calculate_target_stop(price, signal, target_pct=2, stop_pct=1):
    if signal == "BUY":
        target = price * (1 + target_pct/100)
        stop = price * (1 - stop_pct/100)
    elif signal == "SELL":
        target = price * (1 - target_pct/100)
        stop = price * (1 + stop_pct/100)
    else:
        target, stop = None, None
    return target, stop

# ✅ FIXED POSITION SIZING
def calculate_position_size(capital, entry_price, stop_price, risk_percent=1):
    risk_amount = capital * (risk_percent / 100)
    per_share_risk = abs(entry_price - stop_price)

    if per_share_risk < 0.01:
        return 0, 0, 0

    quantity = int(risk_amount / per_share_risk)
    max_affordable_qty = int(capital / entry_price)
    quantity = min(quantity, max_affordable_qty)

    position_value = quantity * entry_price
    return quantity, position_value, risk_amount

# ✅ FIXED REALISTIC BACKTEST
def backtest_strategy(series, capital=100000, risk_pct=1):
    balance = capital
    equity_curve = []
    position = 0
    entry_price = 0

    for i in range(1, len(series)):
        price_prev = series.iloc[i-1]
        price_now = series.iloc[i]

        if position == 0 and price_now > price_prev:
            risk_amount = balance * (risk_pct / 100)
            stop = price_now * 0.99
            qty = int(risk_amount / abs(price_now - stop))

            if qty > 0:
                position = qty
                entry_price = price_now
                balance -= qty * price_now

        elif position > 0 and price_now < price_prev:
            balance += position * price_now
            position = 0

        total_equity = balance + position * price_now
        equity_curve.append(total_equity)

    equity_curve = pd.Series(equity_curve)

    total_return = (equity_curve.iloc[-1] / capital - 1) * 100
    win_rate = (equity_curve.pct_change().fillna(0) > 0).mean() * 100
    drawdown = (equity_curve / equity_curve.cummax() - 1).min() * 100

    return equity_curve, total_return, win_rate, drawdown

# =========================
# ✅ SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙ Trading Control Panel")
    file = st.file_uploader("📂 Upload Market CSV", type=["csv"])
    timeframe = st.selectbox("⏱ Timeframe", ["Original", "1 Min", "5 Min", "15 Min", "30 Min"])
    horizon = st.selectbox("📅 Forecast Horizon", [7, 15, 30, 60])
    capital = st.number_input("💰 Trading Capital", 1000, 10_000_000, 100000, step=1000)
    risk_percent = st.slider("⚠ Risk % per Trade", 0.5, 5.0, 1.0, step=0.5)

if not file:
    st.info("Upload a CSV file to begin.")
    st.stop()

# =========================
# ✅ LOAD DATA (FIXED PRICE COLUMN)
# =========================
df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]

possible_price_cols = ["close", "adj close", "price", "last"]
price_col = None
for col in df.columns:
    if col.lower() in possible_price_cols:
        price_col = col
        break
if price_col is None:
    price_col = df.select_dtypes(include="number").columns[0]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, price_col])
df.set_index(date_col, inplace=True)

tf_map = {"Original": None, "1 Min": "1T", "5 Min": "5T", "15 Min": "15T", "30 Min": "30T"}
if tf_map[timeframe]:
    df = df.resample(tf_map[timeframe]).last().dropna()

series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.2)

st.markdown("<div class='glass-card'><h3>📈 Price Trend</h3></div>", unsafe_allow_html=True)
st.line_chart(series.tail(300))

# =========================
# ✅ RUN ALL MODELS
# =========================
if st.button("🚀 Run AI Models"):

    preds = {}
    metrics = {}
    future_index = pd.date_range(series.index[-1], periods=horizon+1, freq="D")[1:]

    arima = train_arima(train, (5,1,0))
    preds["ARIMA"] = pd.Series(forecast_arima(arima, horizon), index=future_index)

    sarima = train_sarima(train)
    preds["SARIMA"] = pd.Series(forecast_sarima(sarima, horizon), index=future_index)

    prophet = train_prophet(train)
    preds["Prophet"] = pd.Series(forecast_prophet(prophet, horizon).values, index=future_index)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    train_scaled = scaled[:len(train)]

    lstm_model = train_lstm(train_scaled, 20, 2, 8)
    preds["LSTM"] = pd.Series(
        forecast_lstm(lstm_model, scaled, scaler, 20, horizon),
        index=future_index
    )

    # ✅ TRADING SIGNAL + POSITION SIZING
    last_price = series.iloc[-1]
    future_price = preds["ARIMA"].iloc[-1]

    signal, strength = generate_signal(last_price, future_price)
    target, stop = calculate_target_stop(last_price, signal)

    qty, pos_value, risk_amt = calculate_position_size(capital, last_price, stop, risk_percent)

    st.subheader("📢 Trade Signal")
    st.metric("Signal", signal)
    st.metric("Expected Move %", f"{strength:.2f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"{last_price:.2f}")
    c2.metric("Target", f"{target:.2f}")
    c3.metric("Stop", f"{stop:.2f}")
    c4.metric("Quantity", qty)

    # ✅ BACKTEST
    st.subheader("📊 Strategy Backtesting")
    equity, total_return, win_rate, drawdown = backtest_strategy(test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return %", f"{total_return:.2f}%")
    col2.metric("Win Rate %", f"{win_rate:.2f}%")
    col3.metric("Max Drawdown %", f"{drawdown:.2f}%")

    st.line_chart(equity)
