import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# =========================
# ✅ IMPORT YOUR MODELS
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
# ✅ BUY / SELL SIGNAL LOGIC
# =========================
def generate_signal(last_price, forecast_price):
    change_pct = ((forecast_price - last_price) / last_price) * 100

    if change_pct > 2:
        return "✅ BUY", change_pct
    elif change_pct < -2:
        return "🔻 SELL", change_pct
    else:
        return "⏸ HOLD", change_pct

# =========================
# ✅ BASIC UI
# =========================
st.set_page_config(page_title="Stock Forecasting", layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #050a1a, #0b132b);
    color: #dbe7ff;
}
.block-container {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 0 15px rgba(0,200,255,0.15);
}
.stButton button {
    background: linear-gradient(135deg,#0077ff,#00d4ff);
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Forecasting Dashboard (Professional Version)")

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
    if len(nums) == 0:
        return None
    return nums[-1]

def plot_series(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# =========================
# ✅ TAB
# =========================
tab1 = st.tabs(["📂 CSV Forecasting"])[0]

# =========================
# ✅ TAB 1 — FULL FORECASTING
# =========================
with tab1:
    st.subheader("📂 Upload CSV for Forecasting")

    file = st.file_uploader("Upload your stock CSV", type=["csv"])

    if not file:
        st.info("Upload a CSV to begin.")
        st.stop()

    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    st.write("✅ Detected Columns:", list(df.columns))

    date_col = detect_date_column(df)
    price_col = detect_price_column(df)

    if price_col is None:
        st.error("❌ No numeric price column found.")
        st.stop()

    st.success(f"✅ Using Date: `{date_col}` | Price: `{price_col}`")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df.set_index(date_col, inplace=True)

    series = df[price_col].astype(float)

    st.line_chart(series.tail(300))

    series = prepare_series(df, price_col)
    train, test = train_test_split_series(series, 0.2)

    # =========================
    # ✅ FORECAST HORIZON
    # =========================
    horizon = st.selectbox(
        "📅 Select Forecast Horizon (Days)",
        [7, 15, 30, 60],
        index=2
    )

    if st.button("🚀 Run Forecast Models"):

        preds = {}
        metrics = {}
        errors = {}

        # ============ ARIMA ============
        try:
            arima = train_arima(train, order=(5,1,0))
            pred = forecast_arima(arima, horizon)

            future_index = pd.date_range(
                start=series.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq="D"
            )

            preds["ARIMA"] = pd.Series(pred, index=future_index)
            st.success("✅ ARIMA success")
        except Exception as e:
            errors["ARIMA"] = str(e)

        # ============ SARIMA ============
        try:
            sarima = train_sarima(train)
            pred = forecast_sarima(sarima, horizon)
            preds["SARIMA"] = pd.Series(pred, index=future_index)
            st.success("✅ SARIMA success")
        except Exception as e:
            errors["SARIMA"] = str(e)

        # ============ PROPHET ============
        try:
            prophet = train_prophet(train)
            pvals = forecast_prophet(prophet, horizon)
            preds["Prophet"] = pd.Series(pvals.values, index=future_index)
            st.success("✅ Prophet success")
        except Exception as e:
            errors["Prophet"] = str(e)

        # ============ LSTM (CLOUD SAFE) ============
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            train_scaled = scaled[:len(train)]

            lstm_model = train_lstm(
                train_scaled,
                seq_len=20,
                epochs=2,
                batch_size=8
            )

            lvals = forecast_lstm(lstm_model, scaled, scaler, 20, horizon)
            preds["LSTM"] = pd.Series(lvals, index=future_index)
            st.success("✅ LSTM success")
        except Exception as e:
            errors["LSTM"] = str(e)

        if len(preds) == 0:
            st.error("❌ ALL MODELS FAILED")
            st.code(errors)
            st.stop()

        # =========================
        # ✅ INDIVIDUAL FORECASTS
        # =========================
        st.subheader("📊 Individual Forecasts")

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
            align_len = min(len(test), len(p))
            rmse = np.sqrt(mean_squared_error(test.values[:align_len], p.values[:align_len]))
            mse = mean_squared_error(test.values[:align_len], p.values[:align_len])
            mape = np.mean(np.abs((test.values[:align_len] - p.values[:align_len]) / test.values[:align_len])) * 100

            metrics[name] = {
                "RMSE": rmse,
                "MSE": mse,
                "MAPE (%)": mape
            }

        metrics_df = pd.DataFrame(metrics).T
        metrics_df["Rank"] = metrics_df["RMSE"].rank()

        st.dataframe(metrics_df.style.background_gradient(cmap="Blues"))

        best_model = metrics_df.sort_values("RMSE").index[0]
        st.success(f"🏆 Best Model: {best_model}")

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
        ax.set_title("Combined Forecast Comparison")

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
        # ✅ BUY / SELL SIGNAL
        # =========================
        if "ARIMA" in preds:
            last_price = series.iloc[-1]
            future_price = preds["ARIMA"].iloc[-1]

            signal, strength = generate_signal(last_price, future_price)

            st.markdown("---")
            st.subheader("📢 AI Trading Signal")
            st.metric("Expected Change (%)", f"{strength:.2f}%")

            if "BUY" in signal:
                st.success(f"🚀 {signal} — Strong upward momentum expected")
            elif "SELL" in signal:
                st.error(f"⚠ {signal} — Downward momentum detected")
            else:
                st.warning(f"⏸ {signal} — Market moving sideways")

        # =========================
        # ✅ ERRORS (IF ANY)
        # =========================
        if len(errors) > 0:
            st.subheader("⚠ Model Errors")
            for k, v in errors.items():
                st.code(f"{k} → {v}")
