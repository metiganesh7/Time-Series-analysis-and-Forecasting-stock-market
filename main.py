# =========================================================
# 🚀 PROFESSIONAL AI TRADING & FORECASTING DASHBOARD
# =========================================================

import streamlit as st
import sys, os, io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from nsepython import nsefetch

# =========================
# ✅ PREMIUM UI THEME
# =========================
st.markdown("""
<style>
html, body, .stApp {
    background: linear-gradient(135deg, #050a1a, #0b132b);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: #dbe7ff !important;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.block-container {
    backdrop-filter: blur(14px) saturate(160%);
    background: rgba(255, 255, 255, 0.05);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 0 25px rgba(0, 200, 255, 0.15);
    padding: 2rem !important;
}
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}
.stButton button {
    background: linear-gradient(135deg, #0077ff, #00d4ff);
    border-radius: 10px;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Trading & Forecasting Pro Dashboard")

# =========================
# ✅ TECHNICAL INDICATORS
# =========================

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
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    std = df["CH_CLOSING_PRICE"].rolling(20).std()
    df["BB_upper"] = df["EMA_20"] + 2 * std
    df["BB_lower"] = df["EMA_20"] - 2 * std
    return df

# =========================
# ✅ CANDLE PATTERNS
# =========================

def detect_patterns(df):
    patterns = []
    for i in range(1, len(df)):
        o, c, h, l = df.iloc[i][["CH_OPENING_PRICE","CH_CLOSING_PRICE","CH_TRADE_HIGH_PRICE","CH_TRADE_LOW_PRICE"]]
        prev_o, prev_c = df.iloc[i-1][["CH_OPENING_PRICE","CH_CLOSING_PRICE"]]

        if abs(o - c) <= (h - l) * 0.1:
            patterns.append("Doji")
        elif c > o and prev_c < prev_o:
            patterns.append("Bullish Engulfing")
        elif abs(o - l) < abs(h - c):
            patterns.append("Hammer")
        else:
            patterns.append("None")

    df = df.iloc[1:].copy()
    df["Pattern"] = patterns
    return df

# =========================
# ✅ AI MARKET SUMMARY
# =========================

def ai_summary(df):
    latest = df.iloc[-1]
    rsi = latest["RSI"]

    if rsi > 70:
        sentiment = "Overbought – Possible Correction"
    elif rsi < 30:
        sentiment = "Oversold – Possible Bounce"
    else:
        sentiment = "Neutral"

    trend = "Bullish" if latest["EMA_20"] > latest["SMA_50"] else "Bearish"

    return f"""
✅ **Trend:** {trend}  
✅ **RSI Status:** {sentiment}  
✅ **Momentum:** {"Strong" if abs(latest["MACD"]) > 5 else "Moderate"}  
✅ **AI Verdict:** {"Consider Long Positions" if trend == "Bullish" else "Avoid Fresh Buying"}
"""

# =========================
# ✅ NSE DATA FETCH
# =========================

def fetch_nse(symbol):
    api_url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from=06-12-2024&to=06-12-2025"
    data = nsefetch(api_url)
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["CH_TIMESTAMP"])
    df.set_index("date", inplace=True)
    return df

# =========================
# ✅ MULTI-PAGE DASHBOARD
# =========================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Chart + Indicators",
    "🔎 Stock Screener",
    "🕯 Candle Patterns",
    "🤖 AI Market Summary"
])

# =========================
# ✅ TAB 1 — LIVE CHART + INDICATORS
# =========================

with tab1:
    symbol = st.text_input("Enter NSE Symbol", "HDFCBANK").upper().replace(".NS","")
    df = fetch_nse(symbol)
    df = add_indicators(df)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["CH_OPENING_PRICE"],
        high=df["CH_TRADE_HIGH_PRICE"],
        low=df["CH_TRADE_LOW_PRICE"],
        close=df["CH_CLOSING_PRICE"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 RSI & MACD")
    st.line_chart(df[["RSI","MACD","MACD_signal"]])

# =========================
# ✅ TAB 2 — NSE STOCK SCREENER
# =========================

with tab2:
    st.subheader("🔥 NSE Stock Screener – Top Gainers & Losers")

    gainers = nsefetch("https://www.nseindia.com/api/live-analysis-variations?index=gainers")["data"]
    losers = nsefetch("https://www.nseindia.com/api/live-analysis-variations?index=losers")["data"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Top Gainers")
        st.dataframe(pd.DataFrame(gainers)[["symbol","ltp","netPrice"]])

    with col2:
        st.subheader("🔻 Top Losers")
        st.dataframe(pd.DataFrame(losers)[["symbol","ltp","netPrice"]])

# =========================
# ✅ TAB 3 — CANDLE PATTERNS
# =========================

with tab3:
    dfp = detect_patterns(df)
    pattern_count = dfp["Pattern"].value_counts()

    st.subheader("🕯 Candle Pattern Detection")
    st.dataframe(dfp[["CH_CLOSING_PRICE","Pattern"]].tail(20))
    st.bar_chart(pattern_count)

# =========================
# ✅ TAB 4 — AI MARKET SUMMARY
# =========================

with tab4:
    st.subheader("🤖 AI Trading Recommendation")
    st.markdown(ai_summary(df))
