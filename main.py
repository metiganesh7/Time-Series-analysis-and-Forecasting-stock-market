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
# PARTICLE BACKGROUND (STYLE A)
st.markdown("""
<canvas id="particles"></canvas>

<style>
#particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -5;
    pointer-events: none;
}
</style>

<script>
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');

let particlesArray;

let mouse = {
    x: null,
    y: null,
    radius: 120
};

window.addEventListener('mousemove', function (event) {
    mouse.x = event.x;
    mouse.y = event.y;
});

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener('resize', function () {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    init();
});

class Particle {
    constructor(x, y, directionX, directionY, size, color) {
        this.x = x;
        this.y = y;
        this.directionX = directionX;
        this.directionY = directionY;
        this.size = size;
        this.color = color;
    }
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
        ctx.fillStyle = "#00eaff";
        ctx.shadowColor = "#00eaff";
        ctx.shadowBlur = 15;
        ctx.fill();
    }
    update() {
        if (this.x > canvas.width || this.x < 0) { this.directionX = -this.directionX; }
        if (this.y > canvas.height || this.y < 0) { this.directionY = -this.directionY; }

        this.x += this.directionX;
        this.y += this.directionY;

        // Mouse collision effect
        let dx = mouse.x - this.x;
        let dy = mouse.y - this.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < mouse.radius) {
            this.x -= dx / distance;
            this.y -= dy / distance;
        }
        this.draw();
    }
}

function init() {
    particlesArray = [];
    let numberOfParticles = 120;

    for (let i = 0; i < numberOfParticles; i++) {
        let size = (Math.random() * 2) + 1;
        let x = Math.random() * (innerWidth - size * 2);
        let y = Math.random() * (innerHeight - size * 2);
        let directionX = (Math.random() * 0.4) - 0.2;
        let directionY = (Math.random() * 0.4) - 0.2;
        let color = "#00eaff";

        particlesArray.push(new Particle(x, y, directionX, directionY, size, color));
    }
}

function connect() {
    let opacity = 1;
    for (let a = 0; a < particlesArray.length; a++) {
        for (let b = a; b < particlesArray.length; b++) {
            let dx = particlesArray[a].x - particlesArray[b].x;
            let dy = particlesArray[a].y - particlesArray[b].y;
            let dist = dx * dx + dy * dy;

            if (dist < (canvas.width / 20) * (canvas.height / 20)) {
                opacity = 0.1;
                ctx.strokeStyle = "rgba(0, 234, 255," + opacity + ")";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                ctx.stroke();
            }
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < particlesArray.length; i++) {
        particlesArray[i].update();
    }
    connect();
}

init();
animate();

</script>
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
