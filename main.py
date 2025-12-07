import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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

# Prophet & LSTM (SAFE IMPORT)
try:
    from scripts.prophet_model import train_prophet, forecast_prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

try:
    from scripts.lstm_model import train_lstm, forecast_lstm
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Trading Terminal Pro", layout="wide")

# =========================
# ✅ PREMIUM UI
# =========================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #020617, #02030f); color: #e5e7eb; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #020617, #02030f); }
.glass { background: rgba(255,255,255,0.06); border-radius: 18px; padding: 18px; margin-bottom: 20px; }
h1,h2,h3 { background: linear-gradient(90deg,#38bdf8,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stButton button {background: linear-gradient(135deg,#2563eb,#7c3aed) !important; color:white !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='glass'><h1>📊 AI Trading Terminal Pro</h1></div>", unsafe_allow_html=True)

# =========================
# ✅ HELPERS
# =========================
def generate_signal(last_price, forecast_price):
    pct = ((forecast_price - last_price) / last_price) * 100
    if pct > 1: return "BUY", pct
    elif pct < -1: return "SELL", pct
    return "HOLD", pct

def calculate_target_stop(price, signal):
    if signal == "BUY": return price * 1.02, price * 0.99
    if signal == "SELL": return price * 0.98, price * 1.01
    return None, None

def position_size(capital, entry, stop):
    risk_amt = capital * 0.01
    per_share = abs(entry - stop) if stop else 0
    if per_share == 0: return 0
    qty = int(risk_amt / per_share)
    return min(qty, int(capital / entry))

def backtest_engine(series, capital=100000):
    balance = capital
    equity = []
    position = 0

    for i in range(1, len(series)):
        p0, p1 = series.iloc[i-1], series.iloc[i]

        if position == 0 and p1 > p0:
            stop = p1 * 0.99
            qty = int((balance * 0.01) / abs(p1 - stop))
            position = qty
            balance -= qty * p1

        elif position > 0 and p1 < p0:
            balance += position * p1
            position = 0

        equity.append(balance + position * p1)

    equity = pd.Series(equity)
    total_return = (equity.iloc[-1] / capital - 1) * 100
    win_rate = (equity.pct_change().fillna(0) > 0).mean() * 100
    drawdown = (equity / equity.cummax() - 1).min() * 100

    return equity, total_return, win_rate, drawdown

# =========================
# ✅ SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙ Control Panel")
    file = st.file_uploader("📂 Upload CSV", type=["csv"])
    horizon = st.selectbox("📅 Forecast Horizon", [7, 15, 30])
    capital = st.number_input("💰 Capital", 10000, 10_000_000, 100000)
    use_prophet = st.checkbox("Enable Prophet", value=True)
    use_lstm = st.checkbox("Enable LSTM", value=True)

if not file:
    st.stop()

# =========================
# ✅ LOAD DATA
# =========================
df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
price_col = [c for c in df.columns if c.lower() in ["close","adj close","price","last"]]
price_col = price_col[0] if price_col else df.select_dtypes(include="number").columns[0]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, price_col])
df.set_index(date_col, inplace=True)

series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.2)

# =========================
# ✅ TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🎯 Signals", "📊 Backtesting", "📥 Downloads"])

with tab1:
    st.subheader("📈 Price Trend")
    st.line_chart(series.tail(300))

    if st.button("🚀 Run All Models"):
        preds = {}
        metrics = {}
        errors = {}

        future_index = pd.date_range(series.index[-1], periods=horizon+1, freq="D")[1:]

        # ARIMA
        arima = train_arima(train, order=(5,1,0))
        preds["ARIMA"] = pd.Series(forecast_arima(arima, horizon), index=future_index)

        # SARIMA
        sarima = train_sarima(train)
        preds["SARIMA"] = pd.Series(forecast_sarima(sarima, horizon), index=future_index)

        # PROPHET
        if use_prophet and PROPHET_AVAILABLE:
            prophet_model = train_prophet(train)
            pf = forecast_prophet(prophet_model, horizon).values
            preds["Prophet"] = pd.Series(pf, index=future_index)

        # LSTM
        if use_lstm and LSTM_AVAILABLE:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            train_scaled = scaled[:len(train)]

            lstm_model = train_lstm(train_scaled, seq_len=20, epochs=3, batch_size=8)
            lstm_forecast = forecast_lstm(lstm_model, scaled, scaler, 20, horizon)

            preds["LSTM"] = pd.Series(lstm_forecast, index=future_index)

        # Metrics
        for name, p in preds.items():
            align = min(len(test), len(p))
            rmse = np.sqrt(mean_squared_error(test.values[:align], p.values[:align]))
            mse = mean_squared_error(test.values[:align], p.values[:align])
            mape = np.mean(np.abs((test.values[:align] - p.values[:align]) / test.values[:align])) * 100

            metrics[name] = {"RMSE": rmse, "MSE": mse, "MAPE (%)": mape}

        metrics_df = pd.DataFrame(metrics).T
        metrics_df["Rank"] = metrics_df["RMSE"].rank()

        st.session_state["preds"] = preds
        st.session_state["metrics"] = metrics_df

        # ✅ Combined Chart
        fig, ax = plt.subplots(figsize=(12,5))
        series.tail(200).plot(ax=ax, label="Actual")
        for name,p in preds.items(): p.plot(ax=ax, label=name)
        ax.legend()
        st.pyplot(fig)

        st.dataframe(metrics_df.sort_values("Rank"))

with tab2:
    if "preds" in st.session_state:
        preds = st.session_state["preds"]
        metrics_df = st.session_state["metrics"]

        best_model = metrics_df.sort_values("Rank").index[0]
        best_forecast = preds[best_model]

        last_price = series.iloc[-1]
        future_price = best_forecast.iloc[-1]

        signal, strength = generate_signal(last_price, future_price)
        target, stop = calculate_target_stop(last_price, signal)
        qty = position_size(capital, last_price, stop)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best Model", best_model)
        c2.metric("Signal", signal)
        c3.metric("Entry", f"{last_price:.2f}")
        c4.metric("Quantity", qty)

        st.metric("Target", target)
        st.metric("Stop", stop)
        st.metric("Expected Move %", f"{strength:.2f}%")

with tab3:
    equity, total_return, win_rate, drawdown = backtest_engine(test, capital)

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Return %", f"{total_return:.2f}")
    c2.metric("Win Rate %", f"{win_rate:.2f}")
    c3.metric("Max Drawdown %", f"{drawdown:.2f}")

    st.line_chart(equity)

with tab4:
    if "preds" in st.session_state:
        for name,p in st.session_state["preds"].items():
            csv = p.to_csv().encode("utf-8")
            st.download_button(f"⬇ Download {name}", csv, f"{name}_forecast.csv")
