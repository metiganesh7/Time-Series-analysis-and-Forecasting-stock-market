import sys
import os
import io
import datetime as dt

# Ensure scripts folder importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import Model Scripts
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.prophet_model import train_prophet, forecast_prophet
from scripts.lstm_model import train_lstm, forecast_lstm

# UI and Processing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# --------------------------
# Metric Functions
# --------------------------
def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def MAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# --------------------------
# Helper Functions
# --------------------------
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return df.columns[0]  # fallback

def detect_price_column(df):
    candidates = ["Close", "Adj Close", "close", "price", "Close Price"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include="number").columns
    return numeric[-1]

def plot_series(train, test, pred, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    train.plot(ax=ax, label="Train")
    test.plot(ax=ax, label="Test")
    pred.plot(ax=ax, label="Prediction")
    ax.set_title(title)
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting Dashboard (Upload CSV Only)")

st.sidebar.header("📁 Upload CSV File")
uploaded = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

if not uploaded:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load CSV
df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

# Identify Columns
date_col = detect_date_column(df)
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
series = df[price_col]
series.name = price_col

# Display Preview
with st.expander("📊 Dataset Preview", expanded=True):
    st.dataframe(df.tail())

    fig, ax = plt.subplots(figsize=(10, 3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

# Prepare Time Series
series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, test_size=0.2)


# --------------------------
# Sidebar Model Selection
# --------------------------
st.sidebar.header("🧠 Select Models")
models = st.sidebar.multiselect(
    "Choose forecasting models:",
    ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    default=["ARIMA", "SARIMA", "Prophet", "LSTM"]
)

st.sidebar.markdown("---")

# Hyperparameters
arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))
sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

lstm_seq = st.sidebar.number_input("LSTM Sequence Length", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("LSTM Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("LSTM Batch Size", 1, 256, 32)

run_btn = st.sidebar.button("Run Models")

# Storage for all results
combined_predictions = {}
model_scores = {}

col1, col2 = st.columns(2)

# --------------------------
# Run Selected Models
# --------------------------

if run_btn:

    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            pred = pd.Series(forecast_arima(
                train_arima(train.squeeze(), order=arima_order),
                len(test)
            ), index=test.index)

            combined_predictions["ARIMA"] = pred
            model_scores["ARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred),
            }

            col1.subheader("ARIMA Forecast")
            col1.image(plot_series(train, test, pred, "ARIMA Forecast"))


    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            pred = pd.Series(
                forecast_sarima(
                    train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order),
                    len(test)
                ),
                index=test.index
            )

            combined_predictions["SARIMA"] = pred
            model_scores["SARIMA"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred),
            }

            col1.subheader("SARIMA Forecast")
            col1.image(plot_series(train, test, pred, "SARIMA Forecast"))


    # Prophet
    if "Prophet" in models:
        with st.spinner("Running Prophet..."):
            prophet_model = train_prophet(train.squeeze())
            pred = forecast_prophet(prophet_model, len(test))
            pred = pred.reindex(test.index)

            combined_predictions["Prophet"] = pred
            model_scores["Prophet"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred),
            }

            col2.subheader("Prophet Forecast")
            col2.image(plot_series(train, test, pred, "Prophet Forecast"))


    # LSTM
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled)*0.8)
            train_scaled = scaled[:split]

            lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            pred = forecast_lstm(lstm_model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
            pred = pd.Series(pred, index=test.index)

            combined_predictions["LSTM"] = pred
            model_scores["LSTM"] = {
                "RMSE": RMSE(test, pred),
                "MSE": MSE(test, pred),
                "MAPE": MAPE(test, pred),
            }

            col2.subheader("LSTM Forecast")
            col2.image(plot_series(train, test, pred, "LSTM Forecast"))


    # ------------------------------
    # Combined Forecast Comparison
    # ------------------------------
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Model Forecast Comparison")

        fig, ax = plt.subplots(figsize=(12, 5))
        train.plot(ax=ax, label="Train", linewidth=2)
        test.plot(ax=ax, label="Test", linewidth=2)

        for model_name, forecast in combined_predictions.items():
            forecast.plot(ax=ax, label=model_name, linewidth=2)

        ax.set_title("Comparison of All Model Forecasts")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------
    # Model Accuracy Metrics + Ranking
    # ------------------------------
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Accuracy Metrics & Rankings")

        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values(by="RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df) + 1)

        st.dataframe(
            metrics_df.style.background_gradient(cmap="Blues").format({
                "RMSE": "{:.4f}",
                "MSE": "{:.4f}",
                "MAPE": "{:.2f}%"
            })
        )

        best_model = metrics_df.index[0]
        st.success(f"🏆 Best Performing Model: **{best_model}**")
