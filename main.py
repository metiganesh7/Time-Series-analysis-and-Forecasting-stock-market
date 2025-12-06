# =========================================================
# ✅ COMPLETE PROFESSIONAL AI TRADING + FORECASTING DASHBOARD
# =========================================================

import streamlit as st
import sys, os, io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from nsepython import nsefetch

# =========================================================
# ✅ PREMIUM UI
# =========================================================
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

st.title("📊 AI Trading & Forecasting Pro Dashboard")

# =========================================================
# ✅ LOAD FORECAST MODELS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# =========================================================
# ✅ UTILS
# =========================================================

def RMSE(a,b): return np.sqrt(mean_squared_error(a,b))
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower(): return c
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close","close","Adj Close","Price"]:
        if c in df.columns: return c
    return df.select_dtypes(include="number").columns[-1]

def plot_series(train,test,pred,title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax,label="Train")
    test.plot(ax=ax,label="Test")
    pred.plot(ax=ax,label="Forecast")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf,format="png")
    buf.seek(0)
    plt.close()
    return buf

# =========================================================
# ✅ SAFE NSE FETCH
# =========================================================

def safe_nse_fetch(url):
    try:
        data = nsefetch(url)
        if isinstance(data, dict): return data
        return {}
    except Exception:
        return {}

def fetch_nse_history(symbol):
    url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from=01-01-2024&to=31-12-2024"
    data = safe_nse_fetch(url)
    raw = data.get("data",[])
    if len(raw)==0: return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["CH_TIMESTAMP"])
    df.set_index("date", inplace=True)
    return df

# =========================================================
# ✅ INDICATORS
# =========================================================

def add_indicators(df):
    df["EMA_20"] = df["CH_CLOSING_PRICE"].ewm(span=20).mean()
    df["SMA_50"] = df["CH_CLOSING_PRICE"].rolling(50).mean()

    delta = df["CH_CLOSING_PRICE"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["CH_CLOSING_PRICE"].ewm(span=12).mean()
    ema26 = df["CH_CLOSING_PRICE"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    return df

def detect_patterns(df):
    patterns=[]
    for i in range(1,len(df)):
        o,c,h,l = df.iloc[i][["CH_OPENING_PRICE","CH_CLOSING_PRICE","CH_TRADE_HIGH_PRICE","CH_TRADE_LOW_PRICE"]]
        if abs(o-c) <= (h-l)*0.1:
            patterns.append("Doji")
        elif c>o:
            patterns.append("Bullish")
        else:
            patterns.append("Bearish")
    df = df.iloc[1:].copy()
    df["Pattern"] = patterns
    return df

def ai_summary(df):
    latest = df.iloc[-1]
    if latest["RSI"] > 70: state="Overbought"
    elif latest["RSI"]<30: state="Oversold"
    else: state="Neutral"
    trend="Bullish" if latest["EMA_20"]>latest["SMA_50"] else "Bearish"
    return f"""
✅ **Trend:** {trend}  
✅ **RSI:** {state}  
✅ **AI Verdict:** {"Consider Buying" if trend=="Bullish" else "Avoid Fresh Buying"}
"""

# =========================================================
# ✅ MULTI-TABS (ALL FEATURES RESTORED)
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Forecasting (CSV)",
    "📈 Live + Indicators",
    "🔎 Screener",
    "🕯 Candle Patterns",
    "🤖 AI Summary"
])

# =========================================================
# ✅ TAB 1 — ADVANCED CSV FORECASTING + ACCURACY + DOWNLOAD
# =========================================================

with tab1:
    st.subheader("📂 Upload CSV for Forecasting")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if not file:
        st.info("Please upload a CSV file first.")
        st.stop()

    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    st.write("✅ Raw Columns Detected:", list(df.columns))

    # ---- Detect Date Column ----
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_col = c
            break

    if date_col is None:
        st.error("❌ No Date column detected in your CSV.")
        st.stop()

    # ---- Detect Price Column ----
    price_col = None
    for c in ["Close", "close", "Adj Close", "Price"]:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        nums = df.select_dtypes(include="number").columns
        if len(nums) == 0:
            st.error("❌ No numeric price column found.")
            st.stop()
        price_col = nums[-1]

    st.success(f"✅ Using → Date: `{date_col}` | Price: `{price_col}`")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df.set_index(date_col, inplace=True)

    series = df[price_col].astype(float)

    st.line_chart(series.tail(300))

    split = int(len(series) * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    # ============================
    # ✅ FORECAST BUTTON
    # ============================
    if st.button("🚀 Run Forecast Models"):

        preds = {}
        errors = {}
        metrics = {}

        # ============================
        # ✅ ARIMA
        # ============================
        try:
            arima = train_arima(train, order=(5,1,0))
            pred = forecast_arima(arima, len(test))
            preds["ARIMA"] = pd.Series(pred, index=test.index)
        except Exception as e:
            errors["ARIMA"] = str(e)

        # ============================
        # ✅ SARIMA
        # ============================
        try:
            sarima = train_sarima(train)
            pred = forecast_sarima(sarima, len(test))
            preds["SARIMA"] = pd.Series(pred, index=test.index)
        except Exception as e:
            errors["SARIMA"] = str(e)

        # ============================
        # ✅ PROPHET
        # ============================
        try:
            prophet = train_prophet(train)
            pvals = forecast_prophet(prophet, len(test))
            preds["Prophet"] = pd.Series(pvals.values, index=test.index)
        except Exception as e:
            errors["Prophet"] = str(e)

        # ============================
        # ✅ LSTM (Cloud Safe)
        # ============================
        try:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            train_scaled = scaled[:split]

            lstm_model = train_lstm(
                train_scaled,
                seq_len=30,
                epochs=3,
                batch_size=16
            )

            lvals = forecast_lstm(lstm_model, scaled, scaler, 30, len(test))
            preds["LSTM"] = pd.Series(lvals, index=test.index)

        except Exception as e:
            errors["LSTM"] = str(e)

        # ============================
        # ✅ INDIVIDUAL FORECAST PLOTS
        # ============================
        if len(preds) == 0:
            st.error("❌ All models failed.")
            st.code(errors)
            st.stop()

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

        # ============================
        # ✅ MODEL ACCURACY TABLE
        # ============================
        st.subheader("📈 Model Accuracy (RMSE, MSE, Ranking)")

        for name, p in preds.items():
            rmse = np.sqrt(mean_squared_error(test, p))
            mse = mean_squared_error(test, p)
            mape = np.mean(np.abs((test - p) / test)) * 100

            metrics[name] = {
                "RMSE": rmse,
                "MSE": mse,
                "MAPE (%)": mape
            }

        metrics_df = pd.DataFrame(metrics).T
        metrics_df["Rank"] = metrics_df["RMSE"].rank()

        st.dataframe(metrics_df.style.background_gradient(cmap="Blues"))

        csv_metrics = metrics_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇ Download Model Accuracy CSV",
            csv_metrics,
            "model_accuracy.csv",
            "text/csv"
        )

        best_model = metrics_df.sort_values("RMSE").index[0]
        st.success(f"🏆 Best Performing Model: {best_model}")

        # ============================
        # ✅ COMBINED FORECAST CHART
        # ============================
        st.subheader("📊 Combined Forecast Chart")

        fig, ax = plt.subplots(figsize=(12,5))
        train.plot(ax=ax, label="Train")
        test.plot(ax=ax, label="Test")

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

        # ============================
        # ✅ SHOW MODEL ERRORS
        # ============================
        if len(errors) > 0:
            st.subheader("⚠ Model Errors")
            for k, v in errors.items():
                st.code(f"{k} ERROR → {v}")


with tab2:
    symbol = st.text_input("Enter NSE Symbol", "TCS").upper().replace(".NS","")
    df = fetch_nse_history(symbol)

    if df.empty:
        st.warning("NSE blocked or no data.")
    else:
        df = add_indicators(df)

        fig = go.Figure(go.Candlestick(
            x=df.index,
            open=df["CH_OPENING_PRICE"],
            high=df["CH_TRADE_HIGH_PRICE"],
            low=df["CH_TRADE_LOW_PRICE"],
            close=df["CH_CLOSING_PRICE"]
        ))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig,use_container_width=True)

        st.line_chart(df[["EMA_20","SMA_50","RSI","MACD"]])

# =========================================================
# ✅ TAB 3 — SAFE SCREENER
# =========================================================

with tab3:
    st.subheader("🔎 NSE Screener")

    gainers = safe_nse_fetch("https://www.nseindia.com/api/live-analysis-variations?index=gainers")
    losers  = safe_nse_fetch("https://www.nseindia.com/api/live-analysis-variations?index=losers")

    g = gainers.get("data",[])
    l = losers.get("data",[])

    if len(g)==0 or len(l)==0:
        st.warning("⚠ NSE Screener temporarily blocked.")
    else:
        st.dataframe(pd.DataFrame(g)[["symbol","ltp","netPrice"]])
        st.dataframe(pd.DataFrame(l)[["symbol","ltp","netPrice"]])

# =========================================================
# ✅ TAB 4 — CANDLE PATTERNS
# =========================================================

with tab4:
    if not df.empty:
        dfp = detect_patterns(df)
        st.dataframe(dfp[["CH_CLOSING_PRICE","Pattern"]].tail(20))
        st.bar_chart(dfp["Pattern"].value_counts())
    else:
        st.info("Load a stock first in Live tab.")

# =========================================================
# ✅ TAB 5 — AI SUMMARY
# =========================================================

with tab5:
    if not df.empty:
        st.markdown(ai_summary(df))
    else:
        st.warning("Load stock data first.")
