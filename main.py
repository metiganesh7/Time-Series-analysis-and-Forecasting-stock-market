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
# ✅ BACKTESTING
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
    return nums[-1] if len(nums) > 0 else None

def plot_series(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.legend()
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# =========================
# ✅ FILE UPLOAD
# =========================
file = st.file_uploader("📂 Upload Stock CSV", type=["csv"])

if not file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
price_col = detect_price_column(df)

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, price_col])
df.set_index(date_col, inplace=True)

series = df[price_col].astype(float)
series = prepare_series(df, price_col)
train, test = train_test_split_series(series, 0.2)

st.line_chart(series.tail(200))

# =========================
# ✅ FORECAST HORIZON
# =========================
horizon = st.selectbox("📅 Forecast Horizon (Days)", [7, 15, 30, 60], index=2)

# =========================
# ✅ RUN ALL MODELS
# =========================
if st.button("🚀 Run All Models"):

    preds = {}
    metrics = {}
    errors = {}

    # ---------- ARIMA ----------
    try:
        arima = train_arima(train, order=(5,1,0))
        future = forecast_arima(arima, horizon)
        future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
        preds["ARIMA"] = pd.Series(future, index=future_index)
        st.success("✅ ARIMA OK")
    except Exception as e:
        errors["ARIMA"] = str(e)

    # ---------- SARIMA ----------
    try:
        sarima = train_sarima(train)
        future = forecast_sarima(sarima, horizon)
        preds["SARIMA"] = pd.Series(future, index=future_index)
        st.success("✅ SARIMA OK")
    except Exception as e:
        errors["SARIMA"] = str(e)

    # ---------- PROPHET ----------
    try:
        prophet = train_prophet(train)
        pvals = forecast_prophet(prophet, horizon)
        preds["Prophet"] = pd.Series(pvals.values, index=future_index)
        st.success("✅ Prophet OK")
    except Exception as e:
        errors["Prophet"] = str(e)

    # ---------- LSTM ----------
    try:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1,1))
        train_scaled = scaled[:len(train)]

        lstm_model = train_lstm(train_scaled, seq_len=20, epochs=2, batch_size=8)
        lvals = forecast_lstm(lstm_model, scaled, scaler, 20, horizon)

        preds["LSTM"] = pd.Series(lvals, index=future_index)
        st.success("✅ LSTM OK")
    except Exception as e:
        errors["LSTM"] = str(e)

    if len(preds) == 0:
        st.error("❌ All models failed")
        st.code(errors)
        st.stop()

    # =========================
    # ✅ INDIVIDUAL FORECASTS
    # =========================
    st.subheader("📊 Individual Model Forecasts")

    for name, p in preds.items():
        img = plot_series(train, test, p, name)
        st.image(img)
        st.download_button(
            f"⬇ Download {name} Forecast",
            img.getvalue(),
            f"{name}_forecast.png",
            "image/png"
        )

    # =========================
    # ✅ MODEL ACCURACY
    # =========================
    st.subheader("📈 Model Accuracy")

    for name, p in preds.items():
        align = min(len(test), len(p))
        rmse = np.sqrt(mean_squared_error(test.values[:align], p.values[:align]))
        mse = mean_squared_error(test.values[:align], p.values[:align])
        mape = np.mean(np.abs((test.values[:align] - p.values[:align]) / test.values[:align])) * 100

        metrics[name] = {
            "RMSE": rmse,
            "MSE": mse,
            "MAPE (%)": mape
        }

    metrics_df = pd.DataFrame(metrics).T
    metrics_df["Rank"] = metrics_df["RMSE"].rank()
    st.dataframe(metrics_df)

    csv_metrics = metrics_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇ Download Model Accuracy CSV",
        csv_metrics,
        "model_accuracy.csv",
        "text/csv"
    )

    # =========================
    # ✅ COMBINED FORECAST
    # =========================
    st.subheader("📊 Combined Forecast")

    fig, ax = plt.subplots(figsize=(12,5))
    series.plot(ax=ax, label="Actual")

    for name, p in preds.items():
        p.plot(ax=ax, label=name)

    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    st.image(buf)
    st.download_button(
        "⬇ Download Combined Forecast",
        buf.getvalue(),
        "combined_forecast.png",
        "image/png"
    )

    # =========================
    # ✅ BUY / SELL + TARGET / STOP
    # =========================
    if "ARIMA" in preds:
        last_price = series.iloc[-1]
        future_price = preds["ARIMA"].iloc[-1]

        signal, strength = generate_signal(last_price, future_price)
        target, stop = calculate_target_stop(last_price, signal)

        st.subheader("📢 Trading Signal")
        st.metric("Expected Change %", f"{strength:.2f}%")
        st.metric("Signal", signal)

        if target:
            col1, col2, col3 = st.columns(3)
            col1.metric("Current", f"{last_price:.2f}")
            col2.metric("Target", f"{target:.2f}")
            col3.metric("Stop Loss", f"{stop:.2f}")

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
    # ✅ MODEL ERRORS
    # =========================
    if len(errors) > 0:
        st.subheader("⚠ Model Errors")
        for k, v in errors.items():
            st.code(f"{k} → {v}")
