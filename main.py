# app.py (FULL replacement)
import sys
import os
import io
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model scripts (must exist)
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Try to import plotly; if not available, we'll fallback to matplotlib visuals
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY_AVAILABLE = False

# Try to make plotly to_image available (kaleido). We'll test later when exporting.
KALeIDO_AVAILABLE = True
if PLOTLY_AVAILABLE:
    try:
        # test to_image
        fig_test = go.Figure()
        fig_test.to_image(format="png")
    except Exception:
        KALeIDO_AVAILABLE = False

# --------------------------
# THEME + RELIABLE ANIMATED BACKGROUND (CSS ONLY)
# --------------------------
st.set_page_config(page_title="Forecasting + Groww Charts", layout="wide")
st.markdown("""
<style>
/* Base app look */
.stApp {
    position: relative;
    overflow: hidden;
    background: radial-gradient(circle at 10% 10%, #061226 0%, #02040a 35%, #02030a 100%);
    color: #e6eef8;
    font-family: "Poppins", sans-serif;
}

/* Particle-like layered animated blobs (CSS-only, reliable) */
.stApp::before{
  content: "";
  position: fixed;
  top: -10%;
  left: -10%;
  width: 220%;
  height: 220%;
  z-index: -3;
  background-image:
    radial-gradient(circle, rgba(0,230,255,0.10) 1px, transparent 1px),
    radial-gradient(circle, rgba(0,180,255,0.07) 1px, transparent 1px),
    radial-gradient(circle, rgba(180,240,255,0.03) 1px, transparent 1px);
  background-size: 120px 120px, 80px 80px, 50px 50px;
  animation: particleMove 30s linear infinite;
  opacity: 0.9;
  pointer-events: none;
}

.stApp::after{
  content: "";
  position: fixed;
  top: -10%;
  left: -10%;
  width: 220%;
  height: 220%;
  z-index: -2;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,0.01) 0%, transparent 40%),
    radial-gradient(circle at 30% 20%, rgba(0,255,200,0.02), transparent 10%);
  background-size: 400px 400px;
  animation: drift 60s linear infinite;
  opacity: 0.7;
  pointer-events: none;
}

@keyframes particleMove { 0%{transform:translate(0,0)} 50%{transform:translate(-60px,-30px)} 100%{transform:translate(0,0)} }
@keyframes drift { 0%{transform:translate(0,0)} 50%{transform:translate(-80px,-40px)} 100%{transform:translate(0,0)} }

/* grid depth */
.stApp .grid {
  position: fixed; inset:0; z-index:-1;
  background-image:
    linear-gradient(rgba(0,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,255,0.02) 1px, transparent 1px);
  background-size: 200px 200px;
  animation: gridScroll 40s linear infinite;
  opacity: 0.45;
}
@keyframes gridScroll { 0%{transform:translateY(0)} 50%{transform:translateY(-60px)} 100%{transform:translateY(0)} }

/* readable cards */
.block-container {
  backdrop-filter: blur(6px) saturate(1.1);
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.03);
}

/* buttons */
.stButton button { background: linear-gradient(135deg,#6a11cb,#00d4ff); color:white; padding:10px 16px; border-radius:10px; border:none; font-weight:600;}
.stDownloadButton button { background: linear-gradient(135deg,#ffbe0b,#fb5607); color:black; padding:8px 14px; border-radius:10px; border:none; font-weight:700;}

/* small screens */
@media (max-width:600px){ .stApp::before, .stApp::after { display:none; } }
</style>
<div class="grid"></div>
""", unsafe_allow_html=True)

# --------------------------
# Metric functions
# --------------------------
def RMSE(actual, predicted):
    return float(np.sqrt(mean_squared_error(np.array(actual), np.array(predicted))))

def MSE(actual, predicted):
    return float(mean_squared_error(np.array(actual), np.array(predicted)))

def MAPE(actual, predicted):
    actual = np.array(actual).astype(float)
    predicted = np.array(predicted).astype(float)
    mask = actual == 0
    if mask.any():
        actual = actual.copy()
        actual[mask] = 1e-8
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)

# --------------------------
# Helpers
# --------------------------
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]

def detect_price_column(df):
    candidates = ["Close","Adj Close","close","price","Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("No numeric columns found")
    return numeric[-1]

def plot_series_buf(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Forecast")
    ax.set_title(title)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_combined_chart_buf(train, test, combined_predictions):
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)
    for name, fc in combined_predictions.items():
        try:
            fc.plot(ax=ax, label=name, linewidth=2)
        except Exception:
            ax.plot(fc.index, fc.values, label=name, linewidth=2)
    ax.set_title("Model Comparison Chart")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_radar_chart_buf(metrics_df):
    df = metrics_df[["RMSE","MSE","MAPE"]].copy()
    norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        vals = df[col].values.astype(float)
        mn, mx = vals.min(), vals.max()
        if mx - mn == 0:
            norm[col] = 1.0
        else:
            norm[col] = (mx - vals) / (mx - mn)
    labels = list(norm.columns)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    for idx in norm.index:
        values = norm.loc[idx].tolist()
        values += values[:1]
        ax.plot(angles, values, label=idx, linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    ax.set_title("Model Radar Chart (Higher = Better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# Groww-style charts (plotly if available; fallback to matplotlib)
# --------------------------
def groww_charts_figure(df, price_col, chart_type="Line"):
    df = df.copy()
    df["Date"] = df.index
    if PLOTLY_AVAILABLE:
        if chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df[price_col], mode="lines",
                                     line=dict(color="#00E6FF", width=2),
                                     hovertemplate="Price: %{y:.2f}<extra></extra>"))
            fig.update_layout(template="plotly_dark", title="Groww-Style Neon Line Chart",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        if chart_type == "Area":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df[price_col], mode="lines",
                                     line=dict(color="#00B4FF", width=2),
                                     fill="tozeroy", fillcolor="rgba(0,180,255,0.2)",
                                     hovertemplate="Price: %{y:.2f}<extra></extra>"))
            fig.update_layout(template="plotly_dark", title="Groww-Style Gradient Area Chart",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
        if chart_type == "Candlestick":
            required = ["Open","High","Low","Close"]
            if not all(col in df.columns for col in required):
                return None
            fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                                 low=df["Low"], close=df["Close"],
                                                 increasing_line_color="#00E676", decreasing_line_color="#FF1744")])
            fig.update_layout(template="plotly_dark", title="Groww-Style Candlestick",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return fig
    else:
        # fallback: create simple matplotlib figure and return it as BytesIO buffer (not a plotly fig)
        if chart_type in ("Line", "Area"):
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(df.index, df[price_col], color="#00E6FF", linewidth=2)
            if chart_type == "Area":
                ax.fill_between(df.index, df[price_col], color="#00B4FF", alpha=0.2)
            ax.set_title(f"Groww-Style {chart_type} Chart (matplotlib fallback)")
            ax.grid(False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            return buf
        else:
            return None

# --------------------------
# Streamlit UI
# --------------------------
st.title("📈 Stock Forecasting Dashboard with Groww-style Charts")

st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Upload CSV file (must include Date + Price)", type=["csv"])
if not uploaded:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# detect date and price
try:
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
except Exception as e:
    st.error(f"Date column error: {e}")
    st.stop()

try:
    price_col = detect_price_column(df)
except Exception as e:
    st.error(f"Price column error: {e}")
    st.stop()

# Data preview
with st.expander("📊 Data Preview", expanded=True):
    st.dataframe(df.tail())
    fig_preview, ax_preview = plt.subplots(figsize=(10,3))
    df[price_col].plot(ax=ax_preview)
    ax_preview.set_title("Price Series")
    st.pyplot(fig_preview)
    plt.close(fig_preview)

# Groww charts selector and display (placed before model runs so user can inspect data)
st.markdown("---")
st.subheader("📊 Groww-Style Interactive Charts")

chart_type = st.selectbox("Choose chart type", ["Line", "Area", "Candlestick"])
groww_fig = groww_charts_figure(df, price_col, chart_type)

# show interactive plotly if available, else show png fallback
if PLOTLY_AVAILABLE and groww_fig is not None:
    st.plotly_chart(groww_fig, use_container_width=True)
    # plot download (only if plotly->kaleido works)
    if KALeIDO_AVAILABLE:
        try:
            png_bytes = groww_fig.to_image(format="png")
            st.download_button("Download Chart (PNG)", png_bytes, file_name=f"groww_{chart_type.lower()}.png", mime="image/png")
        except Exception:
            st.info("Chart PNG export not available (kaleido missing). You can still interact with the chart.")
    else:
        st.info("Plotly is available but image export is not (kaleido missing). Interactive chart shown.")
elif groww_fig is not None:
    # matplotlib buffer returned
    st.image(groww_fig)
    st.download_button("Download Chart (PNG)", groww_fig.getvalue(), file_name=f"groww_{chart_type.lower()}.png", mime="image/png")
else:
    st.warning("Selected chart type requires columns not present in CSV (candlestick needs Open,High,Low,Close).")

# Prepare timeseries for modeling
series = df[price_col].copy()
series.name = price_col

# convert to daily series and split
series_ts = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series_ts, test_size=0.2)

# Sidebar: models and params
st.sidebar.header("Models & Parameters")
models = st.sidebar.multiselect("Choose models", ["ARIMA","SARIMA","Prophet","LSTM"],
                                default=["ARIMA","SARIMA","Prophet","LSTM"])

def parse_tuple_input(s, length=3, default=(1,1,1)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        return tuple(parts[:length]) if len(parts)>=length else tuple((parts + list(default))[:length])
    except Exception:
        return default

arima_order = parse_tuple_input(st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0"), default=(5,1,0))
sarima_order = parse_tuple_input(st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1"), default=(1,1,1))
seasonal_text = st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12")
try:
    seasonal_order = tuple([int(x.strip()) for x in seasonal_text.split(",")][:4])
    if len(seasonal_order) < 4:
        seasonal_order = (1,1,1,12)
except Exception:
    seasonal_order = (1,1,1,12)

lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)

run = st.sidebar.button("Run Models")

combined_predictions = {}
model_scores = {}

col1, col2 = st.columns(2)

# Run models
if run:
    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Training ARIMA..."):
            try:
                arima_res = train_arima(train.squeeze(), order=arima_order)
                arima_vals = forecast_arima(arima_res, steps=len(test))
                arima_pred = pd.Series(arima_vals, index=test.index)
                combined_predictions["ARIMA"] = arima_pred
                model_scores["ARIMA"] = {"RMSE": RMSE(test, arima_pred), "MSE": MSE(test, arima_pred), "MAPE": MAPE(test, arima_pred)}
                buf = plot_series_buf(train, test, arima_pred, "ARIMA Forecast")
                col1.subheader("ARIMA")
                col1.image(buf)
                # download CSV
                csv_buf = arima_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8")
                col1.download_button("Download ARIMA CSV", csv_buf, file_name="arima_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"ARIMA error: {e}")

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Training SARIMA..."):
            try:
                sarima_res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
                sarima_vals = forecast_sarima(sarima_res, steps=len(test))
                sarima_pred = pd.Series(sarima_vals, index=test.index)
                combined_predictions["SARIMA"] = sarima_pred
                model_scores["SARIMA"] = {"RMSE": RMSE(test, sarima_pred), "MSE": MSE(test, sarima_pred), "MAPE": MAPE(test, sarima_pred)}
                buf = plot_series_buf(train, test, sarima_pred, "SARIMA Forecast")
                col1.subheader("SARIMA")
                col1.image(buf)
                csv_buf = sarima_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8")
                col1.download_button("Download SARIMA CSV", csv_buf, file_name="sarima_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"SARIMA error: {e}")

    # Prophet
    if "Prophet" in models:
        with st.spinner("Training Prophet..."):
            try:
                prophet_input = train.squeeze()
                prophet_model = train_prophet(prophet_input)
                prophet_pred_series = forecast_prophet(prophet_model, periods=len(test))
                prophet_pred_series = prophet_pred_series.reindex(test.index)
                prophet_pred = pd.Series(prophet_pred_series.values, index=test.index)
                combined_predictions["Prophet"] = prophet_pred
                model_scores["Prophet"] = {"RMSE": RMSE(test, prophet_pred), "MSE": MSE(test, prophet_pred), "MAPE": MAPE(test, prophet_pred)}
                buf = plot_series_buf(train, test, prophet_pred, "Prophet Forecast")
                col2.subheader("Prophet")
                col2.image(buf)
                csv_buf = prophet_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8")
                col2.download_button("Download Prophet CSV", csv_buf, file_name="prophet_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prophet error: {e}")

    # LSTM
    if "LSTM" in models:
        with st.spinner("Training LSTM..."):
            try:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(series.values.reshape(-1,1))
                split = int(len(scaled)*0.8)
                train_scaled = scaled[:split]
                lstm_model = train_lstm(train_scaled, seq_len=int(lstm_seq), epochs=int(lstm_epochs), batch_size=int(lstm_batch))
                lstm_vals = forecast_lstm(lstm_model, scaled, scaler, seq_len=int(lstm_seq), steps=len(test))
                lstm_pred = pd.Series(lstm_vals, index=test.index)
                combined_predictions["LSTM"] = lstm_pred
                model_scores["LSTM"] = {"RMSE": RMSE(test, lstm_pred), "MSE": MSE(test, lstm_pred), "MAPE": MAPE(test, lstm_pred)}
                buf = plot_series_buf(train, test, lstm_pred, "LSTM Forecast")
                col2.subheader("LSTM")
                col2.image(buf)
                csv_buf = lstm_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8")
                col2.download_button("Download LSTM CSV", csv_buf, file_name="lstm_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"LSTM error: {e}")

    # Combined chart + CSV download
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Forecast Comparison")
        combined_buf = plot_combined_chart_buf(train, test, combined_predictions)
        st.image(combined_buf)
        st.download_button("Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_chart.png", mime="image/png")

        combined_df = pd.DataFrame(index=test.index)
        for name, s in combined_predictions.items():
            combined_df[name] = s.values
        combined_df["Actual"] = test.values
        combined_csv = combined_df.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode("utf-8")
        st.download_button("Download Combined Forecasts CSV", combined_csv, file_name="combined_forecasts.csv", mime="text/csv")

    # Metrics + ranking + radar + downloads
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Performance Metrics")
        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values("RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df)+1)
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues").format({
            "RMSE": "{:.4f}",
            "MSE": "{:.4f}",
            "MAPE": "{:.2f}%"
        }))
        best = metrics_df.index[0]
        st.success(f"🏆 Best Model: **{best}**")
        metrics_csv = metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode("utf-8")
        st.download_button("Download Metrics CSV", metrics_csv, file_name="model_metrics.csv", mime="text/csv")

        radar_buf = create_radar_chart_buf(metrics_df)
        st.subheader("📡 Radar Chart (Inverted Metrics)")
        st.image(radar_buf)
        st.download_button("Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")

# End of app.py
