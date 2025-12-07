import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys, os

# =========================
# ✅ IMPORT MODELS (STABLE ONLY)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima

# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Stock Forecasting Pro", layout="wide")

# =========================
# ✅ PREMIUM UI
# =========================
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #020617, #02030f); color: white; }
.glass-card {
    background: rgba(255,255,255,0.06); backdrop-filter: blur(12px);
    border-radius: 18px; padding: 20px; margin-bottom: 20px;
}
h1, h2, h3 {
    background: linear-gradient(90deg,#38bdf8,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='glass-card'><h1>📊 AI Stock Forecasting Dashboard (Stable Mode)</h1></div>", unsafe_allow_html=True)

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

def calculate_target_stop(price, signal):
    if signal == "BUY":
        return price * 1.02, price * 0.99
    elif signal == "SELL":
        return price * 0.98, price * 1.01
    return None, None

def calculate_position_size(capital, entry, stop):
    risk_amt = capital * 0.01
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0
    qty = int(risk_amt / per_share_risk)
    return min(qty, int(capital / entry))

# =========================
# ✅ SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙ Controls")
    file = st.file_uploader("📂 Upload CSV", type=["csv"])
    horizon = st.selectbox("📅 Forecast Horizon", [7, 15, 30])
    capital = st.number_input("💰 Capital", 10_000, 10_000_000, 100_000)

if not file:
    st.stop()

# =========================
# ✅ LOAD DATA (SAFE)
# =========================
df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]

price_candidates = ["close", "adj close", "price", "last"]
price_col = None
for col in df.columns:
    if col.lower() in price_candidates:
        price_col = col
        break
if price_col is None:
    price_col = df.select_dtypes(include="number").columns[0]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, price_col])
df.set_index(date_col, inplace=True)

series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.2)

st.subheader("📈 Price Trend")
st.line_chart(series.tail(300))

# =========================
# ✅ RUN MODELS (SAFE)
# =========================
if st.button("🚀 Run Forecast Models", key="run_models_unique"):

    preds = {}

    future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq="D")[1:]

    # ---------- ARIMA ----------
    try:
        st.write("⏳ Running ARIMA...")
        arima = train_arima(train, order=(5,1,0))
        preds["ARIMA"] = pd.Series(
            forecast_arima(arima, horizon),
            index=future_index
        )
        st.success("✅ ARIMA done")
    except Exception as e:
        st.error(f"ARIMA ERROR → {e}")

    # ---------- SARIMA ----------
    try:
        st.write("⏳ Running SARIMA...")
        sarima = train_sarima(train)
        preds["SARIMA"] = pd.Series(
            forecast_sarima(sarima, horizon),
            index=future_index
        )
        st.success("✅ SARIMA done")
    except Exception as e:
        st.error(f"SARIMA ERROR → {e}")

    if not preds:
        st.error("❌ No model produced output. Stopping.")
        st.stop()

    # =========================
    # ✅ INDIVIDUAL FORECAST GRAPHS
    # =========================
    st.subheader("🎯 Individual Forecasts")

    for name, forecast in preds.items():
        fig, ax = plt.subplots(figsize=(8,4))
        series.tail(200).plot(ax=ax, label="Actual")
        forecast.plot(ax=ax, label=name)
        ax.legend()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.download_button(
            f"⬇ Download {name}",
            buf.getvalue(),
            f"{name}_forecast.png",
            "image/png",
            key=f"dl_{name}"
        )

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
    metrics = {}

    for name, p in preds.items():
        align = min(len(test), len(p))
        rmse = np.sqrt(mean_squared_error(test.values[:align], p.values[:align]))
        metrics[name] = rmse

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["RMSE"])
    metrics_df["Rank"] = metrics_df["RMSE"].rank()

    st.subheader("📈 Model Accuracy & Ranking")
    st.dataframe(metrics_df)

    # =========================
    # ✅ TRADING SIGNAL
    # =========================
    last_price = series.iloc[-1]
    future_price = preds[list(preds.keys())[0]].iloc[-1]

    signal, strength = generate_signal(last_price, future_price)
    target, stop = calculate_target_stop(last_price, signal)
    qty = calculate_position_size(capital, last_price, stop)

    st.subheader("📢 Trade Signal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", signal)
    c2.metric("Entry", f"{last_price:.2f}")
    c3.metric("Target", f"{target:.2f}")
    c4.metric("Quantity", qty)
