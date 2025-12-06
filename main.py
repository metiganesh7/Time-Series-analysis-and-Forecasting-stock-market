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
# ✅ UI + STYLE
# =========================
st.set_page_config(page_title="Stock Forecasting Pro", layout="wide")

st.markdown("""
<style>
body { background-color: #020617; color: white; }
.model-card {
    background: linear-gradient(135deg, #0f172a, #020617);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(0,255,255,0.15);
}
.model-title {
    font-size: 22px;
    font-weight: bold;
    color: #38bdf8;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Professional Stock Forecasting & Trading System")

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
# ✅ CSV UPLOAD
# =========================
file = st.file_uploader("📂 Upload Stock CSV (Daily or Intraday)", type=["csv"])

if not file:
    st.info("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
price_col = df.select_dtypes(include="number").columns[-1]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, price_col])
df.set_index(date_col, inplace=True)

tf_map = {
    "Original": None,
    "1 Min": "1T",
    "5 Min": "5T",
    "15 Min": "15T",
    "30 Min": "30T"
}

tf = st.selectbox("⏱ Timeframe", list(tf_map.keys()))
if tf_map[tf]:
    df = df.resample(tf_map[tf]).last().dropna()

series = df[price_col].astype(float)
series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.2)

st.line_chart(series.tail(200))

# =========================
# ✅ SETTINGS
# =========================
horizon = st.selectbox("📅 Forecast Horizon", [7, 15, 30, 60], index=2)

# =========================
# ✅ RUN MODELS
# =========================
if st.button("🚀 Run All Models + Trading Engine"):

    preds = {}
    metrics = {}
    errors = {}
    future_index = pd.date_range(series.index[-1], periods=horizon+1, freq="D")[1:]

    # ARIMA
    try:
        arima = train_arima(train, order=(5,1,0))
        preds["ARIMA"] = pd.Series(forecast_arima(arima, horizon), index=future_index)
    except Exception as e:
        errors["ARIMA"] = str(e)

    # SARIMA
    try:
        sarima = train_sarima(train)
        preds["SARIMA"] = pd.Series(forecast_sarima(sarima, horizon), index=future_index)
    except Exception as e:
        errors["SARIMA"] = str(e)

    # PROPHET
    try:
        prophet = train_prophet(train)
        preds["Prophet"] = pd.Series(forecast_prophet(prophet, horizon).values, index=future_index)
    except Exception as e:
        errors["Prophet"] = str(e)

    # LSTM
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1))
        train_scaled = scaled[:len(train)]
        lstm_model = train_lstm(train_scaled, seq_len=20, epochs=2, batch_size=8)
        preds["LSTM"] = pd.Series(
            forecast_lstm(lstm_model, scaled, scaler, 20, horizon),
            index=future_index
        )
    except Exception as e:
        errors["LSTM"] = str(e)

    # =========================
    # ✅ INDIVIDUAL COLORFUL CHARTS + DOWNLOAD
    # =========================
    st.subheader("🎯 Individual Model Forecasts")

    model_colors = {
        "ARIMA": "#22c55e",
        "SARIMA": "#06b6d4",
        "Prophet": "#a78bfa",
        "LSTM": "#f97316"
    }

    cols = st.columns(2)
    i = 0

    for model_name, forecast in preds.items():
        with cols[i % 2]:
            st.markdown("<div class='model-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='model-title'>{model_name} Forecast</div>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7,4))
            series.tail(200).plot(ax=ax, label="Actual", color="white")
            forecast.plot(ax=ax, label=model_name, color=model_colors.get(model_name, "cyan"))
            ax.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            st.image(buf)
            st.download_button(
                f"⬇ Download {model_name} Forecast",
                buf.getvalue(),
                f"{model_name.lower()}_forecast.png",
                "image/png",
                key=f"dl_{model_name}"
            )

            st.markdown("</div>", unsafe_allow_html=True)
        i += 1

    # =========================
    # ✅ COMBINED COMPARISON
    # =========================
    st.subheader("📊 Combined Model Forecast Comparison")

    fig, ax = plt.subplots(figsize=(12,5))
    series.tail(200).plot(ax=ax, label="Actual")

    for name, p in preds.items():
        p.plot(ax=ax, label=name)

    ax.legend()
    st.pyplot(fig)

    # =========================
    # ✅ MODEL ACCURACY
    # =========================
    for name, p in preds.items():
        align = min(len(test), len(p))
        rmse = np.sqrt(mean_squared_error(test.values[:align], p.values[:align]))
        mse = mean_squared_error(test.values[:align], p.values[:align])
        mape = np.mean(np.abs((test.values[:align] - p.values[:align]) / test.values[:align])) * 100
        metrics[name] = {"RMSE": rmse, "MSE": mse, "MAPE (%)": mape}

    metrics_df = pd.DataFrame(metrics).T
    metrics_df["Rank"] = metrics_df["RMSE"].rank()
    st.subheader("📈 Model Accuracy")
    st.dataframe(metrics_df)

    # =========================
    # ✅ TRADING ENGINE
    # =========================
    last_price = series.iloc[-1]
    future_price = preds[list(preds.keys())[0]].iloc[-1]
    signal, strength = generate_signal(last_price, future_price)
    target, stop = calculate_target_stop(last_price, signal)

    st.subheader("📢 Trade Signal")
    st.metric("Signal", signal)
    st.metric("Expected Move %", f"{strength:.2f}%")

    if target:
        capital = st.number_input("Trading Capital", 1000, 10_000_000, 100000, step=1000)
        risk_percent = st.slider("Risk % per Trade", 0.5, 5.0, 1.0, step=0.5)

        qty, pos_value, risk_amt = calculate_position_size(
            capital, last_price, stop, risk_percent
        )

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

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return %", f"{total_return:.2f}%")
    col2.metric("Win Rate %", f"{win_rate:.2f}%")
    col3.metric("Max Drawdown %", f"{drawdown:.2f}%")

    st.line_chart(equity)

    # =========================
    # ✅ ERRORS
    # =========================
    if errors:
        st.subheader("⚠ Model Errors")
        for k, v in errors.items():
            st.code(f"{k} → {v}")
