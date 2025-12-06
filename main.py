# main.py  (FULL production-ready dashboard)
import sys
import os
import io
import traceback
import datetime as dt

# Ensure scripts dir is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import streamlit as st
st.set_page_config(page_title="Stock Forecasting + Groww Charts", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Try to import optional libraries and scripts with safe fallbacks
PLOTLY_AVAILABLE = False
KALeIDO_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    # quick kaleido test will be done later when exporting
except Exception:
    PLOTLY_AVAILABLE = False

# Import model helper scripts (wrap in try to avoid crash)
_import_errors = []
train_arima = forecast_arima = None
train_sarima = forecast_sarima = None
train_prophet = forecast_prophet = None
train_lstm = forecast_lstm = None
prepare_series = train_test_split_series = None

try:
    from scripts.utils import prepare_series, train_test_split_series
except Exception as e:
    _import_errors.append(f"scripts.utils import error: {e}")

try:
    from scripts.arima_model import train_arima, forecast_arima
except Exception as e:
    _import_errors.append(f"scripts.arima_model import error: {e}")

try:
    from scripts.sarima_model import train_sarima, forecast_sarima
except Exception as e:
    _import_errors.append(f"scripts.sarima_model import error: {e}")

try:
    from scripts.prophet_model import train_prophet, forecast_prophet
except Exception as e:
    _import_errors.append(f"scripts.prophet_model import error: {e}")

try:
    from scripts.lstm_model import train_lstm, forecast_lstm
except Exception as e:
    _import_errors.append(f"scripts.lstm_model import error: {e}")

# ---------------------------
# Helper metrics / plotting
# ---------------------------
def RMSE(actual, predicted):
    return float(np.sqrt(mean_squared_error(np.array(actual), np.array(predicted))))

def MSE(actual, predicted):
    return float(mean_squared_error(np.array(actual), np.array(predicted))))

def MAPE(actual, predicted):
    actual = np.array(actual).astype(float)
    predicted = np.array(predicted).astype(float)
    mask = actual == 0
    if mask.any():
        actual = actual.copy()
        actual[mask] = 1e-8
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)

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

# ---------------------------
# Groww-style charts helper (Plotly preferred)
# ---------------------------
def groww_chart(df, price_col, chart_type="Line"):
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
        # fallback to matplotlib image buffer for line/area
        if chart_type in ("Line","Area"):
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(df.index, df[price_col], color="#00E6FF", linewidth=2)
            if chart_type == "Area":
                ax.fill_between(df.index, df[price_col], color="#00B4FF", alpha=0.2)
            ax.set_title(f"Groww-Style {chart_type} Chart (fallback)")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            return buf
        else:
            return None

# ---------------------------
# UI: theme CSS (reliable CSS-only background)
# ---------------------------
st.markdown("""
<style>
/* Dark glass + animated blobs (CSS only) */
.stApp {
  position: relative;
  overflow: hidden;
  background: radial-gradient(circle at 10% 10%, #061226 0%, #02040a 35%, #02030a 100%);
  color: #e6eef8;
  font-family: "Poppins", sans-serif;
}
.stApp::before{
  content: "";
  position: fixed;
  top: -10%; left:-10%;
  width:220%; height:220%;
  z-index:-3;
  background-image:
    radial-gradient(circle, rgba(0,230,255,0.10) 1px, transparent 1px),
    radial-gradient(circle, rgba(0,180,255,0.07) 1px, transparent 1px),
    radial-gradient(circle, rgba(180,240,255,0.03) 1px, transparent 1px);
  background-size:120px 120px,80px 80px,50px 50px;
  animation: particleMove 30s linear infinite;
  opacity:0.9; pointer-events:none;
}
.stApp::after{
  content: "";
  position: fixed;
  top: -10%; left:-10%;
  width:220%; height:220%;
  z-index:-2;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,0.01) 0%, transparent 40%),
    radial-gradient(circle at 30% 20%, rgba(0,255,200,0.02), transparent 10%);
  background-size:400px 400px;
  animation:drift 60s linear infinite;
  opacity:0.7; pointer-events:none;
}
@keyframes particleMove { 0%{transform:translate(0,0)}50%{transform:translate(-60px,-30px)}100%{transform:translate(0,0)} }
@keyframes drift {0%{transform:translate(0,0)}50%{transform:translate(-80px,-40px)}100%{transform:translate(0,0)}}
.block-container { backdrop-filter: blur(6px) saturate(1.1); background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:14px; border:1px solid rgba(255,255,255,0.03); }
.stButton button { background: linear-gradient(135deg,#6a11cb,#00d4ff); color:white; padding:10px 16px; border-radius:10px; border:none; font-weight:600;}
.stDownloadButton button { background: linear-gradient(135deg,#ffbe0b,#fb5607); color:black; padding:8px 14px; border-radius:10px; border:none; font-weight:700;}
@media (max-width:600px){ .stApp::before, .stApp::after { display:none; } }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Top-level UI and import warnings
# ---------------------------
st.title("📈 Stock Forecasting Dashboard — Groww-style charts + Model comparison")

if _import_errors:
    st.warning("Some script imports failed. Models may be unavailable. See below.")
    for err in _import_errors:
        st.text(err)

# Sidebar: choose data source
st.sidebar.header("Data source")
data_option = st.sidebar.radio("Choose data source:", ("Upload CSV", "Use repository CSV", "YFinance"))

uploaded_file = None
repo_csv_choice = None
ticker_input = None

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
elif data_option == "Use repository CSV":
    # list csv files in repo root
    csv_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".csv")]
    repo_csv_choice = st.sidebar.selectbox("Repository CSV", ["-- choose --"] + csv_files)
else:
    ticker_input = st.sidebar.text_input("Ticker (e.g. ADANIPORTS.NS)", value="ADANIPORTS.NS")

# Additional controls
st.sidebar.markdown("---")
st.sidebar.header("Chart / Model Settings")
chart_choice = st.sidebar.selectbox("Groww chart type", ["Line","Area","Candlestick"])
plotly_export_msg = ""
if PLOTLY_AVAILABLE:
    st.sidebar.write("Plotly detected")
else:
    st.sidebar.write("Plotly not available — using matplotlib fallbacks")

# Read data based on selection
df = None
if data_option == "Upload CSV":
    if not uploaded_file:
        st.info("Upload a CSV (or choose repo CSV or YFinance) to proceed.")
        st.stop()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif data_option == "Use repository CSV":
    if repo_csv_choice in (None, "-- choose --"):
        st.info("Select a repository CSV file.")
        st.stop()
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, repo_csv_choice))
    except Exception as e:
        st.error(f"Failed to read repository CSV: {e}")
        st.stop()
else:  # YFinance
    try:
        import yfinance as yf
        if not ticker_input:
            st.info("Enter a ticker to load from yfinance.")
            st.stop()
        df = yf.download(ticker_input, progress=False)
        if df.empty:
            st.error("YFinance returned no data for that ticker.")
            st.stop()
        # yfinance returns DatetimeIndex; convert to DataFrame with index as Date
        df = df.reset_index()
    except Exception as e:
        st.error(f"YFinance error: {e}")
        st.stop()

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Detect date and price columns
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    # if index looks like dates (e.g., downloaded from yfinance)
    try:
        pd.to_datetime(df.iloc[:,0])
        return df.columns[0]
    except Exception:
        return None

def detect_price_column(df):
    candidates = ["Close","Adj Close","close","Adj Close**","Close*","Close Price","Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include=[np.number]).columns
    return numeric[-1] if len(numeric)>0 else None

date_col = detect_date_column(df)
if date_col is None:
    st.error("Could not detect a date/time column in the CSV.")
    st.stop()

# ensure date col is datetime
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error(f"Failed to parse date column '{date_col}'.")
    st.stop()

df = df.set_index(date_col).sort_index()

price_col = detect_price_column(df)
if price_col is None:
    st.error("Could not find a price/numeric column in the CSV.")
    st.stop()

# Quick preview
with st.expander("Data preview", expanded=True):
    st.dataframe(df.tail(5))
    fig_p, axp = plt.subplots(figsize=(10,3))
    df[price_col].plot(ax=axp)
    axp.set_title("Price series")
    st.pyplot(fig_p)
    plt.close(fig_p)

# Show Groww chart
st.markdown("---")
st.subheader("📊 Groww-style Chart")
groww_fig = groww_chart(df, price_col, chart_choice)
if PLOTLY_AVAILABLE and groww_fig is not None:
    st.plotly_chart(groww_fig, use_container_width=True)
    # attempt to enable png export if kaleido present
    if PLOTLY_AVAILABLE:
        try:
            _ = go.Figure().to_image(format="png")
            KALeIDO_AVAILABLE = True
        except Exception:
            KALeIDO_AVAILABLE = False
    if KALeIDO_AVAILABLE:
        try:
            png = groww_fig.to_image(format="png")
            st.download_button("Download Chart (PNG)", png, file_name=f"groww_{chart_choice.lower()}.png", mime="image/png")
        except Exception:
            st.info("Kaleido not available for PNG export. Interactive chart still works.")
elif groww_fig is not None:
    # matplotlib buffer returned
    st.image(groww_fig)
    st.download_button("Download Chart (PNG)", groww_fig.getvalue(), file_name=f"groww_{chart_choice.lower()}.png", mime="image/png")
else:
    st.warning("Selected chart not available (candlestick needs O/H/L/C columns).")

# Prepare series for modeling
try:
    series = df[price_col].asfreq("D")  # resample/align to daily if needed
    series = series.fillna(method="ffill").fillna(method="bfill")
except Exception as e:
    st.error(f"Failed to prepare series: {e}")
    st.stop()

# Train/test split
def train_test_split_series(s, test_size=0.2):
    n = len(s)
    split = int(n*(1-test_size))
    train = s.iloc[:split]
    test = s.iloc[split:]
    return train, test

train_series, test_series = train_test_split_series(series, test_size=0.2)

# Sidebar: model params
st.sidebar.markdown("---")
st.sidebar.header("Model parameters")
models_selected = st.sidebar.multiselect("Select models", ["ARIMA","SARIMA","Prophet","LSTM"], default=["ARIMA","SARIMA","Prophet","LSTM"])
arima_text = st.sidebar.text_input("ARIMA p,d,q", "5,1,0")
sarima_text = st.sidebar.text_input("SARIMA p,d,q", "1,1,1")
seasonal_text = st.sidebar.text_input("Seasonal P,D,Q,s", "1,1,1,12")
lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)
run_models_btn = st.sidebar.button("Run models 🚀")

def parse_tuple(s, length=3, default=(1,1,1)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) >= length:
            return tuple(parts[:length])
        else:
            return tuple((parts + list(default))[:length])
    except Exception:
        return default

arima_order = parse_tuple(arima_text, 3, (5,1,0))
sarima_order = parse_tuple(sarima_text, 3, (1,1,1))
try:
    seasonal_order = tuple(int(x.strip()) for x in seasonal_text.split(",")[:4])
    if len(seasonal_order) < 4:
        seasonal_order = (1,1,1,12)
except Exception:
    seasonal_order = (1,1,1,12)

# Containers for outputs
combined_predictions = {}
model_scores = {}
col1, col2 = st.columns(2)

# Run models when user clicks
if run_models_btn:
    # ARIMA
    if "ARIMA" in models_selected and train_arima and forecast_arima:
        with st.spinner("Training ARIMA..."):
            try:
                arima_res = train_arima(train_series, order=arima_order, save_path="models")
                arima_vals = forecast_arima(arima_res, steps=len(test_series))
                arima_pred = pd.Series(arima_vals, index=test_series.index)
                combined_predictions["ARIMA"] = arima_pred
                model_scores["ARIMA"] = {"RMSE": RMSE(test_series, arima_pred), "MSE": MSE(test_series, arima_pred), "MAPE": MAPE(test_series, arima_pred)}
                buf = plot_series_buf(train_series, test_series, arima_pred, "ARIMA Forecast")
                col1.subheader("ARIMA Forecast")
                col1.image(buf)
                col1.download_button("Download ARIMA CSV", arima_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8"), file_name="arima_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"ARIMA failed: {e}\n{traceback.format_exc()}")

    # SARIMA
    if "SARIMA" in models_selected and train_sarima and forecast_sarima:
        with st.spinner("Training SARIMA..."):
            try:
                sarima_res = train_sarima(train_series, order=sarima_order, seasonal_order=seasonal_order, save_path="models")
                sarima_vals = forecast_sarima(sarima_res, steps=len(test_series))
                sarima_pred = pd.Series(sarima_vals, index=test_series.index)
                combined_predictions["SARIMA"] = sarima_pred
                model_scores["SARIMA"] = {"RMSE": RMSE(test_series, sarima_pred), "MSE": MSE(test_series, sarima_pred), "MAPE": MAPE(test_series, sarima_pred)}
                buf = plot_series_buf(train_series, test_series, sarima_pred, "SARIMA Forecast")
                col1.subheader("SARIMA Forecast")
                col1.image(buf)
                col1.download_button("Download SARIMA CSV", sarima_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8"), file_name="sarima_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"SARIMA failed: {e}\n{traceback.format_exc()}")

    # Prophet
    if "Prophet" in models_selected and train_prophet and forecast_prophet:
        with st.spinner("Training Prophet..."):
            try:
                prophet_model = train_prophet(train_series, save_path="models")
                prophet_series = forecast_prophet(prophet_model, periods=len(test_series))
                prophet_series = prophet_series.reindex(test_series.index)
                prophet_pred = pd.Series(prophet_series.values, index=test_series.index)
                combined_predictions["Prophet"] = prophet_pred
                model_scores["Prophet"] = {"RMSE": RMSE(test_series, prophet_pred), "MSE": MSE(test_series, prophet_pred), "MAPE": MAPE(test_series, prophet_pred)}
                buf = plot_series_buf(train_series, test_series, prophet_pred, "Prophet Forecast")
                col2.subheader("Prophet Forecast")
                col2.image(buf)
                col2.download_button("Download Prophet CSV", prophet_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8"), file_name="prophet_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prophet failed: {e}\n{traceback.format_exc()}")

    # LSTM
    if "LSTM" in models_selected and train_lstm and forecast_lstm:
        with st.spinner("Training LSTM..."):
            try:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(series.values.reshape(-1,1))
                split_idx = int(len(scaled)*0.8)
                train_scaled = scaled[:split_idx]
                lstm_model = train_lstm(train_scaled, seq_len=int(lstm_seq), epochs=int(lstm_epochs), batch_size=int(lstm_batch), save_path="models")
                lstm_vals = forecast_lstm(lstm_model, scaled, scaler, seq_len=int(lstm_seq), steps=len(test_series))
                lstm_pred = pd.Series(lstm_vals, index=test_series.index)
                combined_predictions["LSTM"] = lstm_pred
                model_scores["LSTM"] = {"RMSE": RMSE(test_series, lstm_pred), "MSE": MSE(test_series, lstm_pred), "MAPE": MAPE(test_series, lstm_pred)}
                buf = plot_series_buf(train_series, test_series, lstm_pred, "LSTM Forecast")
                col2.subheader("LSTM Forecast")
                col2.image(buf)
                col2.download_button("Download LSTM CSV", lstm_pred.reset_index().rename(columns={price_col:"forecast"}).to_csv(index=False).encode("utf-8"), file_name="lstm_forecast.csv", mime="text/csv")
            except Exception as e:
                st.error(f"LSTM failed: {e}\n{traceback.format_exc()}")

    # Combined chart + downloads
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Forecast Comparison")
        combined_buf = plot_combined_chart_buf(train_series, test_series, combined_predictions)
        st.image(combined_buf)
        st.download_button("Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_chart.png", mime="image/png")
        # combined CSV
        combined_df = pd.DataFrame(index=test_series.index)
        for name, s in combined_predictions.items():
            combined_df[name] = s.values
        combined_df["Actual"] = test_series.values
        st.download_button("Download Combined Forecasts CSV", combined_df.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode("utf-8"), file_name="combined_forecasts.csv", mime="text/csv")

    # Metrics + ranking + radar + downloads
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Performance Metrics")
        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values("RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df)+1)
        st.dataframe(metrics_df.style.background_gradient(cmap="Blues").format({
            "RMSE":"{:.4f}",
            "MSE":"{:.4f}",
            "MAPE":"{:.2f}%"
        }))
        best_model = metrics_df.index[0]
        st.success(f"🏆 Best Model: **{best_model}**")
        st.download_button("Download Metrics CSV", metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode("utf-8"), file_name="metrics.csv", mime="text/csv")
        radar_buf = create_radar_chart_buf(metrics_df)
        st.subheader("📡 Radar Chart (Inverted Metrics)")
        st.image(radar_buf)
        st.download_button("Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")

# end
