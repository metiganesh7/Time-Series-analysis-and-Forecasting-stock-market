import sys
import os
import io
import datetime as dt

# ensure scripts folder is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import model scripts
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

# --------------------------
# APPLY DARK GLASSMORPHISM THEME
# --------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* GLOBAL SETTINGS */
.stApp {
    background: radial-gradient(circle at top left, #0f0f17, #08080f, #000000);
    font-family: 'Poppins', sans-serif;
    color: #f1f1f1;
    animation: fadeIn 1s ease-in-out;
}

/* SMOOTH FADE */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

/* FLOATING CARDS WITH GLOW */
.block-container {
    background: rgba(20, 20, 30, 0.4);
    border-radius: 18px;
    padding: 2rem;
    margin-top: 20px;
    border: 1px solid rgba(100, 100, 255, 0.15);
    backdrop-filter: blur(14px);
    box-shadow: 0 0 25px rgba(0, 122, 255, 0.15),
                0 0 45px rgba(0, 122, 255, 0.10);
    animation: cardFloat 6s ease-in-out infinite alternate;
}

@keyframes cardFloat {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

/* HEADERS */
h1 {
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 42px !important;
    font-weight: 700 !important;
}
h2, h3 {
    color: #e4e4e4 !important;
    text-shadow: 0px 0px 12px rgba(255, 255, 255, 0.15);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 20, 0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] * {
    color: #f5f5f5 !important;
}

/* SIDEBAR TITLE */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 {
    color: #9ae6ff !important;
    text-shadow: 0 0 8px rgba(0,180,255,0.6);
}

/* BUTTONS: CYBER NEON */
.stButton button {
    background: linear-gradient(135deg, #6927ff, #00d4ff);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: 0.25s ease-in-out;
    box-shadow: 0 0 12px rgba(120, 70, 255, 0.6);
}

.stButton button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 0 20px #00d4ff;
}

/* DOWNLOAD BUTTONS */
.stDownloadButton button {
    background: linear-gradient(135deg, #f7b733, #fc4a1a);
    color: #000;
    padding: 10px 16px;
    border-radius: 12px;
    border: none;
    font-weight: 700;
    transition: 0.25s ease-in-out;
}

.stDownloadButton button:hover {
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 0 20px rgba(255,150,40,0.7);
}

/* DATAFRAME TABLE */
table {
    color: #ffffff !important;
}

thead th {
    background: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
}

tbody td {
    background: rgba(255,255,255,0.06) !important;
    color: #cccccc !important;
}

/* IMAGE BORDER GLOW */
img {
    border-radius: 12px;
    box-shadow: 0 0 18px rgba(0,255,255,0.15);
}

/* EXPANDERS */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.1);
    color: #fff !important;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)


# --------------------------
# METRIC FUNCTIONS
# --------------------------
def RMSE(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def MSE(actual, predicted):
    return mean_squared_error(actual, predicted)

def MAPE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual == 0
    if mask.any():
        actual = actual.copy()
        actual[mask] = 1e-8
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# --------------------------
# BASIC HELPERS
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


def plot_combined_chart(train, test, combined_predictions):
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax, label="Train", linewidth=2)
    test.plot(ax=ax, label="Test", linewidth=2)
    for name, fc in combined_predictions.items():
        fc.plot(ax=ax, label=name, linewidth=2)
    ax.set_title("Model Comparison Chart")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def create_radar_chart(metrics_df):
    df = metrics_df[["RMSE","MSE","MAPE"]].copy()
    norm = pd.DataFrame(index=df.index, columns=df.columns)

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
# STREAMLIT UI LAYOUT
# --------------------------
st.title("📈 Stock Price Forecasting Dashboard")

st.sidebar.header("Upload CSV")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if not file:
    st.warning("Upload a CSV to continue.")
    st.stop()

df = pd.read_csv(file)
df.columns = [c.strip() for c in df.columns]

date_col = detect_date_column(df)
df[date_col] = pd.to_datetime(df[date_col])
df.set_index(date_col, inplace=True)

price_col = detect_price_column(df)
series = df[price_col]

with st.expander("📊 Data Preview", True):
    st.dataframe(df.tail())
    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax)
    ax.set_title("Price Series")
    st.pyplot(fig)
    plt.close(fig)

series = prepare_series(df, col=price_col, freq="D")
train, test = train_test_split_series(series, 0.2)

# --------------------------
# SIDEBAR MODEL CONFIG
# --------------------------
st.sidebar.header("Models")

models = st.sidebar.multiselect(
    "Select Models",
    ["ARIMA","SARIMA","Prophet","LSTM"],
    default=["ARIMA","SARIMA","Prophet","LSTM"]
)

arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))

sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

lstm_seq = st.sidebar.number_input("LSTM seq len", 10, 200, 60)
lstm_epochs = st.sidebar.number_input("Epochs", 1, 50, 5)
lstm_batch = st.sidebar.number_input("Batch", 1, 256, 32)

run = st.sidebar.button("Run Models 🚀")

combined_predictions = {}
model_scores = {}

col1, col2 = st.columns(2)

# --------------------------
# RUN MODELS
# --------------------------
if run:

    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            model_res = train_arima(train.squeeze(), order=arima_order)
            pred_vals = forecast_arima(model_res, len(test))
            pred = pd.Series(pred_vals, index=test.index)

            combined_predictions["ARIMA"] = pred
            model_scores["ARIMA"] = {
                "RMSE": RMSE(test,pred),
                "MSE": MSE(test,pred),
                "MAPE": MAPE(test,pred)
            }

            buf = plot_series_buf(train, test, pred, "ARIMA Forecast")
            col1.subheader("ARIMA Forecast")
            col1.image(buf)

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            model_res = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
            pred_vals = forecast_sarima(model_res, len(test))
            pred = pd.Series(pred_vals, index=test.index)

            combined_predictions["SARIMA"] = pred
            model_scores["SARIMA"] = {
                "RMSE": RMSE(test,pred),
                "MSE": MSE(test,pred),
                "MAPE": MAPE(test,pred)
            }

            buf = plot_series_buf(train, test, pred, "SARIMA Forecast")
            col1.subheader("SARIMA Forecast")
            col1.image(buf)

    # PROPHET
    if "Prophet" in models:
        with st.spinner("Running Prophet..."):
            model_p = train_prophet(train.squeeze())
            pred_vals = forecast_prophet(model_p, len(test))
            pred_vals = pred_vals.reindex(test.index)
            pred = pd.Series(pred_vals.values, index=test.index)

            combined_predictions["Prophet"] = pred
            model_scores["Prophet"] = {
                "RMSE": RMSE(test,pred),
                "MSE": MSE(test,pred),
                "MAPE": MAPE(test,pred)
            }

            buf = plot_series_buf(train, test, pred, "Prophet Forecast")
            col2.subheader("Prophet Forecast")
            col2.image(buf)

    # LSTM
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(series.values.reshape(-1,1))
            split = int(len(scaled)*0.8)
            train_scaled = scaled[:split]

            lstm_model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
            pred_vals = forecast_lstm(lstm_model, scaled, scaler, lstm_seq, len(test))
            pred = pd.Series(pred_vals, index=test.index)

            combined_predictions["LSTM"] = pred
            model_scores["LSTM"] = {
                "RMSE": RMSE(test,pred),
                "MSE": MSE(test,pred),
                "MAPE": MAPE(test,pred)
            }

            buf = plot_series_buf(train, test, pred, "LSTM Forecast")
            col2.subheader("LSTM Forecast")
            col2.image(buf)


    # --------------------------
    # COMBINED CHART
    # --------------------------
    if combined_predictions:
        st.markdown("---")
        st.subheader("📊 Combined Forecast Chart")

        combined_buf = plot_combined_chart(train, test, combined_predictions)
        st.image(combined_buf)

        st.download_button(
            "Download Combined Chart (PNG)",
            combined_buf.getvalue(),
            "combined_chart.png",
            "image/png"
        )


    # --------------------------
    # METRICS + RANKING
    # --------------------------
    if model_scores:
        st.markdown("---")
        st.subheader("📈 Model Performance Metrics")

        metrics_df = pd.DataFrame(model_scores).T
        metrics_df = metrics_df.sort_values("RMSE")
        metrics_df["Rank"] = range(1, len(metrics_df)+1)

        st.dataframe(metrics_df.style.background_gradient(cmap="Blues"))

        best = metrics_df.index[0]
        st.success(f"🏆 Best Model: **{best}**")

        metrics_csv = metrics_df.to_csv().encode("utf-8")
        st.download_button("Download Metrics CSV", metrics_csv, "metrics.csv", "text/csv")


        # RADAR CHART
        st.subheader("📡 Radar Chart (Inverted Metrics)")
        radar_buf = create_radar_chart(metrics_df)
        st.image(radar_buf)

        st.download_button(
            "Download Radar Chart (PNG)",
            radar_buf.getvalue(),
            "radar_chart.png",
            "image/png"
        )
