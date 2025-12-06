# main.py — Full app (Background Option 2: Groww-style Blue Gradient)
import sys
import os
import io
import traceback
import datetime as dt

# ensure scripts folder importable
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

# Try optional Plotly
PLOTLY_AVAILABLE = False
KALeIDO_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    # test kaleido availability lazily later during export
except Exception:
    PLOTLY_AVAILABLE = False

# Safe imports of script modules (don't crash at import)
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
# Metrics and helper functions
# ---------------------------
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
# UI: Groww-style blue gradient background + styling
# ---------------------------
st.markdown("""
<style>
/* Groww-style Blue Gradient Background + UI touches */
.stApp {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #071029 0%, #071d2b 40%, #02121a 100%);
  color: #e6eef8;
  font-family: "Poppins", sans-serif;
}

/* subtle diagonal stripes for texture */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(90deg, rgba(255,255,255,0.01) 1px, transparent 1px);
  background-size: 320px 320px;
  z-index: -2;
  opacity: 0.12;
  pointer-events: none;
}

/* soft glowing blobs */
.stApp::after {
  content: "";
  position: fixed;
  top: -15%;
  left: -10%;
  width: 50%;
  height: 70%;
  background: radial-gradient(circle at 20% 20%, rgba(0,230,255,0.10), transparent 40%);
  z-index: -1;
  pointer-events: none;
}

/* content card look */
.block-container {
  backdrop-filter: blur(6px) saturate(1.05);
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.03);
  padding: 12px;
}

/* Buttons */
.stButton button {
  background: linear-gradient(135deg,#00b4ff,#006bff);
  color:white; padding:10px 16px; border-radius:10px; border:none; font-weight:600;
}
.stDownloadButton button {
  background: linear-gradient(135deg,#ffd166,#ff6b6b);
  color:black; padding:8px 14px; border-radius:10px; border:none; font-weight:700;
}

@media (max-width:600px){ .stApp::before, .stApp::after { display:none; } }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Top UI and initial warnings
# ---------------------------
st.title("📈 Stock Forecasting Dashboard — Groww-style Charts")

if _import_errors:
    st.warning("Some imports failed — certain models may be unavailable.")
    for e in _import_errors:
        st.text(e)

# Sidebar - Data source
st.sidebar.header("Data Source")
data_option = st.sidebar.radio("Choose data source:", ("Upload CSV", "Repository CSV", "YFinance"))

uploaded = None
repo_choice = None
ticker = ""

if data_option == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
elif data_option == "Repository CSV":
    csvs = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".csv")]
    repo_choice = st.sidebar.selectbox("Choose CSV from repo", ["-- choose --"] + csvs)
else:
    ticker = st.sidebar.text_input("Ticker (e.g. ADANIPORTS.NS)", value="ADANIPORTS.NS")

# Chart and model settings
st.sidebar.markdown("---")
st.sidebar.header("Chart & Model Settings")
chart_choice = st.sidebar.selectbox("Groww chart type", ["Line","Area","Candlestick"])
st.sidebar.write("Plotly available:" , "Yes" if PLOTLY_AVAILABLE else "No")

# read data based on selection
df = None
if data_option == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV to proceed.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
elif data_option == "Repository CSV":
    if repo_choice is None or repo_choice == "-- choose --":
        st.info("Select a repository CSV file.")
        st.stop()
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, repo_choice))
    except Exception as e:
        st.error(f"Failed to read repository CSV: {e}")
        st.stop()
else:
    try:
        import yfinance as yf
        if not ticker:
            st.info("Enter a ticker to fetch from YFinance.")
            st.stop()
        df = yf.download(ticker, progress=False)
        if df is None or df.empty:
            st.error("YFinance returned no data for that ticker.")
            st.stop()
        # yfinance returns a DataFrame with DatetimeIndex; reset to column so our detection works
        df = df.reset_index()
    except Exception as e:
        st.error(f"YFinance error: {e}")
        st.stop()

# sanitize column names
df.columns = [str(c).strip() for c in df.columns]

# detect date and price columns
def detect_date_col(df_local):
    for c in df_local.columns:
        if "date" in str(c).lower() or "time" in str(c).lower():
            return c
    # fallback: first column that can be parsed as date
    for c in df_local.columns:
        try:
            pd.to_datetime(df_local[c])
            return c
        except Exception:
            continue
    return None

def detect_price_col(df_local):
    candidates = ["Close","Adj Close","close","Adj Close**","Close*","Close Price","Price"]
    for c in candidates:
        if c in df_local.columns:
            return c
    numeric = df_local.select_dtypes(include=[np.number]).columns
    return numeric[-1] if len(numeric)>0 else None

date_col = detect_date_col(df)
if date_col is None:
    st.error("Could not auto-detect a date/time column in the dataset.")
    st.stop()

# parse date column
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error(f"Failed to parse date column '{date_col}'.")
    st.stop()

df = df.set_index(date_col).sort_index()

price_col = detect_price_col(df)
if price_col is None:
    st.error("Could not detect a numeric price column in the dataset.")
    st.stop()

# preview
with st.expander("Data Preview", expanded=True):
    st.dataframe(df.tail(6))
    figp, axp = plt.subplots(figsize=(10,3))
    df[price_col].plot(ax=axp)
    axp.set_title("Price series")
    st.pyplot(figp)
    plt.close(figp)

# show Groww-style chart
st.markdown("---")
st.subheader("📊 Groww-style Interactive Chart")
groww_fig = groww_chart(df, price_col, chart_choice)

if PLOTLY_AVAILABLE and groww_fig is not None:
    st.plotly_chart(groww_fig, use_container_width=True)
    # test kaleido for PNG export
    try:
        test_img = go.Figure().to_image(format="png")
        KALeIDO_AVAILABLE = True
    except Exception:
        KALeIDO_AVAILABLE = False
    if KALeIDO_AVAILABLE:
        try:
            png = groww_fig.to_image(format="png")
            st.download_button("Download Chart (PNG)", png, file_name=f"groww_{chart_choice.lower()}.png", mime="image/png")
        except Exception:
            st.info("Plotly PNG export failed (kaleido). Interactive chart still available.")
    else:
        st.info("Plotly interactive shown. PNG export requires kaleido in runtime.")
elif groww_fig is not None:
    # matplotlib buffer
    st.image(groww_fig)
    st.download_button("Download Chart (PNG)", groww_fig.getvalue(), file_name=f"groww_{chart_choice.lower()}.png", mime="image/png")
else:
    st.warning("Selected chart not available (candlestick needs Open/High/Low/Close).")

# prepare series for modeling (daily)
try:
    series = df[price_col].asfreq("D")
    series = series.fillna(method="ffill").fillna(method="bfill")
except Exception as e:
    st.error(f"Failed to prepare series: {e}")
    st.stop()

# train/test split (simple)
def train_test_split_series_local(s, test_size=0.2):
    n = len(s)
    split = int(n*(1-test_size))
    train = s.iloc[:split]
    test = s.iloc[split:]
    return train, test

train_series, test_series = train_test_split_series_local(series, test_size=0.2)

# Sidebar: modeling params
st.sidebar.markdown("---")
st.sidebar.header("Model parameters")
models_selected = st.sidebar.multiselect("Choose models", ["ARIMA","SARIMA","Prophet","LSTM"], default=["ARIMA","SARIMA","Prophet","LSTM"])
arima_text = st.sidebar.text_input("ARIMA p,d,q", "5,1,0")
sarima_text = st.sidebar.text_input("SARIMA p,d,q", "1,1,1")
seasonal_text = st.sidebar.text_input("Seasonal P,D,Q,s", "1,1,1,12")
lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)
run_btn = st.sidebar.button("Run Models 🚀")

def parse_tuple(s, length=3, default=(1,1,1)):
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) >= length:
            return tuple(parts[:length])
        else:
            padded = list(parts) + list(default)
            return tuple(padded[:length])
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

# containers
combined_predictions = {}
model_scores = {}
col1, col2 = st.columns(2)

# run models
if run_btn:
    # ARIMA
    if "ARIMA" in models_selected:
        if train_arima is None or forecast_arima is None:
            st.error("ARIMA functions not available (import failed).")
        else:
            with st.spinner("Training ARIMA..."):
                try:
                    arima_res = train_arima(train_series.squeeze(), order=arima_order, save_path="models")
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
    if "SARIMA" in models_selected:
        if train_sarima is None or forecast_sarima is None:
            st.error("SARIMA functions not available (import failed).")
        else:
            with st.spinner("Training SARIMA..."):
                try:
                    sarima_res = train_sarima(train_series.squeeze(), order=sarima_order, seasonal_order=seasonal_order, save_path="models")
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
    if "Prophet" in models_selected:
        if train_prophet is None or forecast_prophet is None:
            st.error("Prophet functions not available (import failed).")
        else:
            with st.spinner("Training Prophet..."):
                try:
                    prophet_model = train_prophet(train_series.squeeze(), save_path="models")
                    prophet_series = forecast_prophet(prophet_model, periods=len(test_series))
                    # reindex to test
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
    if "LSTM" in models_selected:
        if train_lstm is None or forecast_lstm is None:
            st.error("LSTM functions not available (import failed).")
        else:
            with st.spinner("Training LSTM..."):
                try:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(series.values.reshape(-1,1))
                    split = int(len(scaled)*0.8)
                    train_scaled = scaled[:split]
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

    # Combined chart + CSV
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Forecast Comparison")
        combined_buf = plot_combined_chart_buf(train_series, test_series, combined_predictions)
        st.image(combined_buf)
        st.download_button("Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_chart.png", mime="image/png")
        combined_df = pd.DataFrame(index=test_series.index)
        for name, s in combined_predictions.items():
            combined_df[name] = s.values
        combined_df["Actual"] = test_series.values
        st.download_button("Download Combined Forecasts CSV", combined_df.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode("utf-8"), file_name="combined_forecasts.csv", mime="text/csv")

    # Metrics + ranking + radar
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
        best = metrics_df.index[0]
        st.success(f"🏆 Best Model: **{best}**")
        st.download_button("Download Metrics CSV", metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode("utf-8"), file_name="metrics.csv", mime="text/csv")
        radar_buf = create_radar_chart_buf(metrics_df)
        st.subheader("📡 Radar Chart (Inverted Metrics)")
        st.image(radar_buf)
        st.download_button("Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")

# end of main.py
