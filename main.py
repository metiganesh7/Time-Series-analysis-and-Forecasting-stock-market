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
}
img { border-radius: 16px; }
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ HEADER
# =========================
st.markdown("""
<div class='glass-card'>
    <h1>📊 AI Stock Forecasting & Trading Dashboard</h1>
    <p style='color:#9ca3af'>Forecasting • Signals • Risk Management • Backtesting</p>
</div>
""", unsafe_allow_html=True)

# =========================
# ✅ HELPERS
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

def calculate_position_size(capital, entry_price, stop_price, risk_percent=1):
    risk_amount = capital * (risk_percent / 100)
    per_share_risk = abs(entry_price - stop_price)
    if per_share_risk < 0.01:
        return 0, 0, 0
    quantity = int(risk_amount / per_share_risk)
    max_qty = int(capital / entry_price)
    quantity = min(quantity, max_qty)
    return quantity, quantity * entry_price, risk_amount

def backtest_strategy(series, capital=100000, risk_pct=1):
    balance = capital
    equity_curve = []
    position = 0

    for i in range(1, len(series)):
        prev_price = series.iloc[i-1]
        price = series.iloc[i]

        if position == 0 and price > prev_price:
            risk_amt = balance * (risk_pct / 100)
            stop = price * 0.99
            qty = int(risk_amt / abs(price - stop))
            if qty > 0:
                position = qty
                balance -= qty * price
        elif position > 0 and price < prev_price:
            balance += position * price
            position = 0

        equity_curve.append(balance + position * price)

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
    file = st.file_uploader("📂 Upload CSV", type=["csv"])
    timeframe = st.selectbox("⏱ Timeframe", ["Original", "1 Min", "5 Min", "15 Min", "30 Min"])
    horizon = st.selectbox("📅 Forecast Horizon", [7, 15, 30, 60])
    capital = st.number_input("💰 Trading Capital", 1000, 10_000_000, 100000, step=1000)
    risk_percent = st.slider("⚠ Risk %", 0.5, 5.0, 1.0, step=0.5)

if not file:
    st.stop()

# =========================
# ✅ LOAD DATA
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

st.line_chart(series.tail(300))

# =========================
# ✅ RUN MODELS (SINGLE BUTTON ONLY)
# =========================
if st.button("🚀 Run AI Models", key="run_models_unique"):

    preds = {}
    metrics = {}
    errors = {}

    future_index = pd.date_range(series.index[-1], periods=horizon+1, freq="D")[1:]

    try:
        arima = train_arima(train, order=(5,1,0))
        preds["ARIMA"] = pd.Series(forecast_arima(arima, horizon), index=future_index)
    except Exception as e:
        errors["ARIMA"] = str(e)

    try:
        sarima = train_sarima(train)
        preds["SARIMA"] = pd.Series(forecast_sarima(sarima, horizon), index=future_index)
    except Exception as e:
        errors["SARIMA"] = str(e)

    try:
        prophet = train_prophet(train)
        pvals = forecast_prophet(prophet, horizon)
        preds["Prophet"] = pd.Series(pvals.values, index=future_index)
    except Exception as e:
        errors["Prophet"] = str(e)

    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1))
        train_scaled = scaled[:len(train)]
        lstm_model = train_lstm(train_scaled, 20, 2, 8)
        lstm_vals = forecast_lstm(lstm_model, scaled, scaler, 20, horizon)
        preds["LSTM"] = pd.Series(lstm_vals, index=future_index)
    except Exception as e:
        errors["LSTM"] = str(e)

    if errors:
        st.subheader("❌ Model Errors")
        for k, v in errors.items():
            st.code(f"{k} → {v}")

    if not preds:
        st.error("No model ran successfully.")
        st.stop()

    st.success("✅ All available models finished!")

    # =========================
    # ✅ INDIVIDUAL GRAPHS
    # =========================
    st.subheader("🎯 Individual Model Forecasts")
    model_colors = {"ARIMA":"#22c55e","SARIMA":"#06b6d4","Prophet":"#a78bfa","LSTM":"#f97316"}
    cols = st.columns(2)
    i = 0

    for model_name, forecast in preds.items():
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(7,4))
            series.tail(200).plot(ax=ax, label="Actual")
            forecast.plot(ax=ax, label=model_name, color=model_colors.get(model_name,"cyan"))
            ax.legend()
            buf = io.BytesIO()
            fig.savefig(buf, format="png"); buf.seek(0)
            st.image(buf)
            st.download_button(
                f"⬇ Download {model_name}",
                buf.getvalue(),
                f"{model_name}_forecast.png",
                "image/png",
                key=f"dl_{model_name}"
            )
        i += 1

    # =========================
    # ✅ COMBINED COMPARISON
    # =========================
    st.subheader("📊 Combined Forecast Comparison")
    fig, ax = plt.subplots(figsize=(12,5))
    series.tail(200).plot(ax=ax, label="Actual")
    for name, p in preds.items():
        p.plot(ax=ax, label=name)
    ax.legend()
    st.pyplot(fig)

    # =========================
    # ✅ MODEL ACCURACY + RANKING
    # =========================
    for name, p in preds.items():
        align = min(len(test), len(p))
        rmse = np.sqrt(mean_squared_error(test.values[:align], p.values[:align]))
        mse = mean_squared_error(test.values[:align], p.values[:align])
        mape = np.mean(np.abs((test.values[:align] - p.values[:align]) / test.values[:align])) * 100
        metrics[name] = {"RMSE": rmse, "MSE": mse, "MAPE (%)": mape}

    metrics_df = pd.DataFrame(metrics).T
    metrics_df["Rank"] = metrics_df["RMSE"].rank()
    st.subheader("📈 Model Accuracy & Rankings")
    st.dataframe(metrics_df)

    # =========================
    # ✅ TRADING SIGNAL + POSITION SIZE
    # =========================
    last_price = series.iloc[-1]
    future_price = preds[list(preds.keys())[0]].iloc[-1]
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

    # =========================
    # ✅ BACKTESTING
    # =========================
    st.subheader("📊 Strategy Backtesting")
    equity, total_return, win_rate, drawdown = backtest_strategy(test)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Return %", f"{total_return:.2f}%")
    c2.metric("Win Rate %", f"{win_rate:.2f}%")
    c3.metric("Max Drawdown %", f"{drawdown:.2f}%")
    st.line_chart(equity)
