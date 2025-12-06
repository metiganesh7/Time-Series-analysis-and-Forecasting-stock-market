# main.py (Optimized, production-ready)
import sys
import os
import io
from typing import Tuple, Dict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Ensure scripts directory is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model utilities (these are local)
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.lstm_model import train_lstm, forecast_lstm

# Try Prophet safely
try:
    from scripts.prophet_model import train_prophet, forecast_prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# --------------------------
# UI THEME (clean & premium)
# --------------------------
st.set_page_config(page_title="Premium Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    html, body, .stApp { background-color: #0A0F1F !important; color: #E4E8F0; font-family: Poppins, sans-serif; }
    .block-container { background: rgba(255,255,255,0.03); border-radius: 16px; padding: 18px; border: 1px solid rgba(255,255,255,0.04);}
    .chart-card { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 14px; border: 1px solid rgba(255,255,255,0.06); }
    .stButton > button { background: linear-gradient(135deg,#6B73FF,#000DFF); color: white; border-radius: 10px; padding: 8px 14px; font-weight:600; }
    .stDownloadButton > button { background: linear-gradient(135deg,#FFB300,#FFDD55); color: black; border-radius: 10px; padding: 8px 12px; font-weight:700; }
    .metric { padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.02); margin-bottom:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Helper metrics & plotting
# --------------------------
def RMSE(a, p): return float(np.sqrt(mean_squared_error(a, p)))
def MSE(a, p): return float(mean_squared_error(a, p))
def MAPE(a, p):
    a = np.array(a, dtype=float)
    p = np.array(p, dtype=float)
    a[a == 0] = 1e-8
    return float(np.mean(np.abs((a - p) / a)) * 100)

def detect_date_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    # fallback to first column
    return df.columns[0]

def detect_price_column(df: pd.DataFrame) -> str:
    candidates = ["Close", "Adj Close", "close", "price", "Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("No numeric column found to act as price.")
    return numeric[-1]

def plot_series_buf(train: pd.Series, test: pd.Series, pred: pd.Series, title: str) -> io.BytesIO:
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

def plot_combined_buf(train: pd.Series, test: pd.Series, preds: Dict[str, pd.Series]) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)
    for name, series in preds.items():
        ax.plot(series.index, series.values, label=name, linewidth=2)
    ax.set_title("Combined Model Comparison")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_radar_buf(metrics_df: pd.DataFrame) -> io.BytesIO:
    # Invert metrics so smaller is better -> normalize to [0,1]
    df = metrics_df[["RMSE","MSE","MAPE"]].astype(float)
    norm = (df.max() - df) / (df.max() - df.min() + 1e-8)
    labels = list(norm.columns)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    for idx in norm.index:
        vals = norm.loc[idx].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=idx, linewidth=2)
        ax.fill(angles, vals, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Model Radar (higher = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# TradingView Dashboard generator
# --------------------------
def tradingview_dashboard(file_name: str, height_chart: int = 560):
    # Build TV symbol guess from filename (support "TICKER.NS.csv" or "HDFC.csv")
    base = os.path.splitext(file_name)[0]
    if "." in base:
        tv_symbol = base.upper()
    else:
        tv_symbol = base.upper() + ".NS"  # default to NSE
    # 1) Ticker tape
    ticker_tape_html = """
    <div class="tradingview-widget-container">
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js">
      {
        "symbols": [{"proName":"NSE:NIFTY","title":"NIFTY"},{"proName":"BSE:SENSEX","title":"SENSEX"},{"proName":"CRYPTO:BTCUSD","title":"BTC"}],
        "colorTheme":"dark",
        "isTransparent":false,
        "displayMode":"adaptive"
      }
      </script>
    </div>
    """
    components.html(ticker_tape_html, height=80)

    # 2) Advanced chart
    chart_html = f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "width":"100%",
        "height":{height_chart},
        "symbol":"{tv_symbol}",
        "interval":"D",
        "timezone":"Etc/UTC",
        "theme":"dark",
        "style":"1",
        "locale":"en",
        "toolbar_bg":"#000000",
        "enable_publishing":false,
        "hide_top_toolbar":false,
        "allow_symbol_change":true,
        "container_id":"tv_chart"
      }});
      </script>
    </div>
    """
    components.html(chart_html, height=height_chart + 20)

    # 3) Market overview
    market_html = """
    <div class="tradingview-widget-container">
    <script src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js">
    {
      "colorTheme":"dark",
      "dateRange":"12M",
      "showChart":true,
      "locale":"en",
      "height":"420",
      "tabs":[
        {"title":"Indices","symbols":[{"s":"NSE:NIFTY"},{"s":"BSE:SENSEX"},{"s":"NASDAQ:NDX"}]},
        {"title":"Crypto","symbols":[{"s":"CRYPTO:BTCUSD"},{"s":"CRYPTO:ETHUSD"}]}
      ]
    }
    </script>
    </div>
    """
    components.html(market_html, height=420)

    # 4) Technical analysis
    ta_html = f"""
    <div class="tradingview-widget-container">
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js">
      {{
        "interval":"1D",
        "width":"100%",
        "isTransparent":false,
        "height":360,
        "symbol":"{tv_symbol}",
        "showIntervalTabs":true,
        "colorTheme":"dark",
        "locale":"en"
      }}
      </script>
    </div>
    """
    components.html(ta_html, height=380)

    # 5) Screener
    screener_html = """
    <div class="tradingview-widget-container">
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js">
      {
        "width":"100%",
        "height":"520",
        "defaultColumn":"overview",
        "defaultScreen":"general",
        "market":"india",
        "showToolbar":true,
        "colorTheme":"dark",
        "locale":"en"
      }
      </script>
    </div>
    """
    st.subheader("📋 TradingView Stock Screener")
    components.html(screener_html, height=540)


# --------------------------
# App layout and logic
# --------------------------
def main():
    st.title("📈 Optimized Forecast Dashboard (Premium UI)")

    # Sidebar uploader
    st.sidebar.header("Upload & Settings")
    uploaded = st.sidebar.file_uploader("Upload CSV (one file)", type=["csv"])
    if uploaded is None:
        st.sidebar.info("Upload a CSV with Date and Price columns (e.g. Close).")
        st.stop()

    # Read csv robustly
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    df.columns = [c.strip() for c in df.columns]

    # detect date & price columns
    try:
        date_col = detect_date_column(df)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().all():
            st.error("Date column conversion failed. Please ensure the file has a valid date column.")
            st.stop()
        df = df.set_index(date_col)
    except Exception as e:
        st.error(f"Date parsing error: {e}")
        st.stop()

    try:
        price_col = detect_price_column(df)
    except Exception as e:
        st.error(f"Price detection error: {e}")
        st.stop()

    series = df[price_col].astype(float)

    # Preview & TradingView
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.tail())
        fig, ax = plt.subplots(figsize=(10, 3))
        series.plot(ax=ax, title=f"{price_col} series")
        st.pyplot(fig)
        plt.close(fig)

    # TradingView dashboard (optimized placement)
    tradingview_dashboard(uploaded.name)

    # Prepare series for modeling
    series_prepared = prepare_series(df, col=price_col, freq="D")
    train, test = train_test_split_series(series_prepared, test_size=0.2)

    # Sidebar model config
    st.sidebar.subheader("Models")
    model_opts = ["ARIMA", "SARIMA", "LSTM"]
    if PROPHET_AVAILABLE:
        model_opts.append("Prophet")
    models = st.sidebar.multiselect("Select models to run", model_opts, default=model_opts)

    # Hyperparameters
    arima_text = st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0")
    sarima_text = st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1")
    seasonal_text = st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12")

    def parse_tuple(s: str, n: int, default: Tuple[int, ...]):
        try:
            parts = [int(x.strip()) for x in s.split(",")]
            return tuple(parts[:n]) if len(parts) >= n else default
        except Exception:
            return default

    arima_order = parse_tuple(arima_text, 3, (5,1,0))
    sarima_order = parse_tuple(sarima_text, 3, (1,1,1))
    seasonal_order = parse_tuple(seasonal_text, 4, (1,1,1,12))

    lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
    lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)

    run = st.sidebar.button("▶ Run Models")

    # Output containers
    combined_predictions = {}
    model_scores = {}

    col1, col2 = st.columns(2)

    if run:
        # ARIMA
        if "ARIMA" in models:
            with st.spinner("Running ARIMA..."):
                try:
                    arima_res = train_arima(train.squeeze(), order=arima_order)
                    arima_pred_vals = forecast_arima(arima_res, steps=len(test))
                    arima_pred = pd.Series(arima_pred_vals, index=test.index)
                    combined_predictions["ARIMA"] = arima_pred
                    model_scores["ARIMA"] = {"RMSE": RMSE(test, arima_pred), "MSE": MSE(test, arima_pred), "MAPE": MAPE(test, arima_pred)}
                    col1.subheader("ARIMA Forecast")
                    col1.image(plot_series_buf(train, test, arima_pred, "ARIMA Forecast"))
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

        # SARIMA
        if "SARIMA" in models:
            with st.spinner("Running SARIMA..."):
                try:
                    sarima_res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
                    sarima_pred_vals = forecast_sarima(sarima_res, steps=len(test))
                    sarima_pred = pd.Series(sarima_pred_vals, index=test.index)
                    combined_predictions["SARIMA"] = sarima_pred
                    model_scores["SARIMA"] = {"RMSE": RMSE(test, sarima_pred), "MSE": MSE(test, sarima_pred), "MAPE": MAPE(test, sarima_pred)}
                    col1.subheader("SARIMA Forecast")
                    col1.image(plot_series_buf(train, test, sarima_pred, "SARIMA Forecast"))
                except Exception as e:
                    st.error(f"SARIMA failed: {e}")

        # Prophet
        if "Prophet" in models and PROPHET_AVAILABLE:
            with st.spinner("Running Prophet..."):
                try:
                    prophet_model = train_prophet(train.squeeze())
                    prophet_pred_series = forecast_prophet(prophet_model, periods=len(test)).reindex(test.index)
                    prophet_pred = pd.Series(prophet_pred_series.values, index=test.index)
                    combined_predictions["Prophet"] = prophet_pred
                    model_scores["Prophet"] = {"RMSE": RMSE(test, prophet_pred), "MSE": MSE(test, prophet_pred), "MAPE": MAPE(test, prophet_pred)}
                    col2.subheader("Prophet Forecast")
                    col2.image(plot_series_buf(train, test, prophet_pred, "Prophet Forecast"))
                except Exception as e:
                    st.error(f"Prophet failed: {e}")
        elif "Prophet" in models and not PROPHET_AVAILABLE:
            st.warning("Prophet is not available in this environment and was skipped.")

        # LSTM
        if "LSTM" in models:
            with st.spinner("Running LSTM..."):
                try:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
                    split_idx = int(len(scaled) * 0.8)
                    train_scaled = scaled[:split_idx]
                    lstm_model = train_lstm(train_scaled, seq_len=int(lstm_seq), epochs=int(lstm_epochs), batch_size=int(lstm_batch))
                    lstm_vals = forecast_lstm(lstm_model, scaled, scaler, seq_len=int(lstm_seq), steps=len(test))
                    lstm_pred = pd.Series(lstm_vals, index=test.index)
                    combined_predictions["LSTM"] = lstm_pred
                    model_scores["LSTM"] = {"RMSE": RMSE(test, lstm_pred), "MSE": MSE(test, lstm_pred), "MAPE": MAPE(test, lstm_pred)}
                    col2.subheader("LSTM Forecast")
                    col2.image(plot_series_buf(train, test, lstm_pred, "LSTM Forecast"))
                except Exception as e:
                    st.error(f"LSTM failed: {e}")

        # Combined chart & download
        if combined_predictions:
            st.markdown("---")
            st.subheader("📊 Combined Forecast Chart")
            combined_buf = plot_combined_buf(train, test, combined_predictions)
            st.image(combined_buf)
            st.download_button("⬇ Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_chart.png", mime="image/png")

            # Combined CSV (actual + forecasts)
            combined_df = pd.DataFrame(index=test.index)
            combined_df["Actual"] = test.values
            for name, series_pred in combined_predictions.items():
                combined_df[name] = series_pred.values
            csv_bytes = combined_df.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Combined Forecasts (CSV)", csv_bytes, file_name="combined_forecasts.csv", mime="text/csv")

        # Metrics, ranking, radar, downloads
        if model_scores:
            st.markdown("---")
            st.subheader("📈 Model Performance Metrics")
            metrics_df = pd.DataFrame(model_scores).T
            metrics_df = metrics_df.sort_values("RMSE")
            metrics_df["Rank"] = range(1, len(metrics_df) + 1)
            st.dataframe(metrics_df.style.format({"RMSE":"{:.4f}","MSE":"{:.4f}","MAPE":"{:.2f}%"}))

            best = metrics_df.index[0]
            st.success(f"🏆 Best Model: {best}")

            metrics_csv = metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Metrics CSV", metrics_csv, file_name="metrics.csv", mime="text/csv")

            radar_buf = create_radar_buf(metrics_df)
            st.subheader("📡 Radar Chart")
            st.image(radar_buf)
            st.download_button("⬇ Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")


if __name__ == "__main__":
    main()
