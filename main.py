import sys
import os
import io
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model scripts (these should exist in scripts/)
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# UI + data libs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --------------------------
# Metric functions
# --------------------------
def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def MAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    # avoid division by zero
    mask = actual == 0
    if mask.any():
        actual = actual.copy()
        actual[mask] = 1e-8
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# --------------------------
# Helpers
# --------------------------
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]

def detect_price_column(df):
    candidates = ["Close", "Adj Close", "close", "price", "Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    if len(numeric) == 0:
        raise ValueError("No numeric column found to use as price.")
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

def plot_combined_chart(train, test, combined_predictions):
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)
    for name, forecast in combined_predictions.items():
        try:
            forecast.plot(ax=ax, label=name, linewidth=2)
        except Exception:
            # fallback: plot with index and values
            ax.plot(forecast.index, forecast.values, label=name)
    ax.set_title("Comparison of Model Forecasts")
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def create_radar_chart(metrics_df):
    """
    Inverted radar chart: lower error => larger area.
    metrics_df must have columns RMSE, MSE, MAPE and index = model names.
    """
    # Normalize per metric to 0..1 then invert so higher is better
    df = metrics_df[["RMSE","MSE","MAPE"]].copy()
    norm = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        col_vals = df[col].values.astype(float)
        mn, mx = col_vals.min(), col_vals.max()
        if mx - mn == 0:
            # all equal -> assign 1.0
            norm[col] = 1.0
        else:
            # invert: lower error -> higher score
            norm[col] = (mx - col_vals) / (mx - mn)
    # radar plotting
    labels = list(norm.columns)
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
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
    ax.set_title("Model Performance (Inverted; higher = better)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Time Series Forecasting + Comparison", layout="wide")
st.title("📈 Time Series Forecasting Dashboard (Upload CSV)")

st.sidebar.header("📁 Upload CSV")
uploaded = st.sidebar.file_uploader("Upload CSV file (Date + Price)", type=["csv"])
if uploaded is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# load df
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Unable to read uploaded CSV: {e}")
    st.stop()

# sanitize column names
df.columns = [c.strip() for c in df.columns]

# detect date + price columns
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

series = df[price_col]
series.name = price_col

# preview
with st.expander("📊 Data Preview", expanded=True):
    st.dataframe(df.tail())
    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

# prepare series
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)

# model controls
st.sidebar.header("🧠 Models & Params")
models = st.sidebar.multiselect("Select models to run", ["ARIMA","SARIMA","Prophet","LSTM"],
                                default=["ARIMA","SARIMA","Prophet","LSTM"])

st.sidebar.markdown("---")
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM batch", 1, 256, 32)

run_btn = st.sidebar.button("Run Models")

# containers for results
combined_predictions = {}
model_scores = {}

col1, col2 = st.columns(2)

# --------------------------
# Run
# --------------------------
if run_btn:
    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Training & forecasting ARIMA..."):
            model_res = train_arima(train.squeeze(), order=arima_order)
            pred_vals = forecast_arima(model_res, steps=len(test))
            pred = pd.Series(pred_vals, index=test.index)
            combined_predictions["ARIMA"] = pred
            model_scores["ARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
            buf = plot_series_buf(train, test, pred, "ARIMA Forecast")
            col1.subheader("ARIMA Forecast")
            col1.image(buf)
            # download per-model forecast CSV
            csv_buf = pred.reset_index().rename(columns={price_col: "forecast"}).to_csv(index=False).encode('utf-8')
            col1.download_button("Download ARIMA CSV", csv_buf, file_name="arima_forecast.csv", mime="text/csv")

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Training & forecasting SARIMA..."):
            model_res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred_vals = forecast_sarima(model_res, steps=len(test))
            pred = pd.Series(pred_vals, index=test.index)
            combined_predictions["SARIMA"] = pred
            model_scores["SARIMA"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
            buf = plot_series_buf(train, test, pred, "SARIMA Forecast")
            col1.subheader("SARIMA Forecast")
            col1.image(buf)
            csv_buf = pred.reset_index().rename(columns={price_col: "forecast"}).to_csv(index=False).encode('utf-8')
            col1.download_button("Download SARIMA CSV", csv_buf, file_name="sarima_forecast.csv", mime="text/csv")

    # Prophet
    if "Prophet" in models:
        with st.spinner("Training & forecasting Prophet..."):
            prophet_input = train.squeeze()
            prophet_model = train_prophet(prophet_input)
            prophet_pred = forecast_prophet(prophet_model, periods=len(test))
            # ensure alignment
            prophet_pred = prophet_pred.reindex(test.index)
            pred = pd.Series(prophet_pred.values, index=test.index)
            combined_predictions["Prophet"] = pred
            model_scores["Prophet"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
            buf = plot_series_buf(train, test, pred, "Prophet Forecast")
            col2.subheader("Prophet Forecast")
            col2.image(buf)
            csv_buf = pred.reset_index().rename(columns={price_col: "forecast"}).to_csv(index=False).encode('utf-8')
            col2.download_button("Download Prophet CSV", csv_buf, file_name="prophet_forecast.csv", mime="text/csv")

    # LSTM
    if "LSTM" in models:
        with st.spinner("Training & forecasting LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled)*0.8)
            train_scaled = scaled[:split]
            lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            pred_vals = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
            pred = pd.Series(pred_vals, index=test.index)
            combined_predictions["LSTM"] = pred
            model_scores["LSTM"] = {"RMSE": RMSE(test, pred), "MSE": MSE(test, pred), "MAPE": MAPE(test, pred)}
            buf = plot_series_buf(train, test, pred, "LSTM Forecast")
            col2.subheader("LSTM Forecast")
            col2.image(buf)
            csv_buf = pred.reset_index().rename(columns={price_col: "forecast"}).to_csv(index=False).encode('utf-8')
            col2.download_button("Download LSTM CSV", csv_buf, file_name="lstm_forecast.csv", mime="text/csv")

    # ------------------------------
    # Combined forecast chart + download
    # ------------------------------
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Model Forecast Comparison")

        combined_buf = plot_combined_chart(train, test, combined_predictions)
        st.image(combined_buf)
        st.download_button("Download Combined Chart (PNG)", combined_buf.getvalue(), file_name="combined_forecast.png", mime="image/png")

        # create combined CSV (index + each model forecast as column)
        combined_df = pd.DataFrame(index=test.index)
        for name, s in combined_predictions.items():
            combined_df[name] = s.values
        # include test (actual) for reference
        combined_df["Actual"] = test.values
        combined_csv = combined_df.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode('utf-8')
        st.download_button("Download Combined Forecasts CSV", combined_csv, file_name="combined_forecasts.csv", mime="text/csv")

    # ------------------------------
    # Metrics table + ranking + downloads
    # ------------------------------
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Accuracy Metrics & Rankings")

        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values(by="RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df) + 1)

        st.dataframe(metrics_df.style.background_gradient(cmap="Blues").format({
            "RMSE": "{:.4f}",
            "MSE": "{:.4f}",
            "MAPE": "{:.2f}%"
        }))

        # metrics CSV download
        metrics_csv = metrics_df.reset_index().rename(columns={"index":"Model"}).to_csv(index=False).encode('utf-8')
        st.download_button("Download Metrics CSV", metrics_csv, file_name="model_metrics.csv", mime="text/csv")

        # Best model
        best_model = metrics_df.index[0]
        st.success(f"🏆 Best Performing Model: **{best_model}** (RMSE: {metrics_df.iloc[0]['RMSE']:.4f})")

        # Radar chart (inverted: lower error => larger area)
        radar_buf = create_radar_chart(metrics_df)
        st.subheader("📡 Inverted Radar Chart (lower error => larger area)")
        st.image(radar_buf)
        st.download_button("Download Radar Chart (PNG)", radar_buf.getvalue(), file_name="radar_chart.png", mime="image/png")
