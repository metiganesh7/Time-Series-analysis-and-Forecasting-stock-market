import sys
import os
import io
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Ensure /scripts import works
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Import forecasting modules
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.lstm_model import train_lstm, forecast_lstm

# Safe Prophet import
try:
    from scripts.prophet_model import train_prophet, forecast_prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False


# ------------------------------------------------------------
# UI THEME
# ------------------------------------------------------------
st.set_page_config(page_title="Forecast Dashboard", layout="wide")

st.markdown("""
<style>
html, body, .stApp { background-color: #0A0F1F; color: #E4E8F0; font-family: Poppins, sans-serif; }
.block-container { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 14px; }
.stButton > button { background: linear-gradient(135deg,#6B73FF,#000DFF); color:white; padding:10px 18px;
                     border-radius:10px; font-weight:600; }
.stDownloadButton > button { background: linear-gradient(135deg,#FFB300,#FFDD55); color:black;
                             padding:10px 14px; border-radius:8px; font-weight:700; }
.chart-card { background:rgba(255,255,255,0.04); padding:15px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)



# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def RMSE(a,p): return float(np.sqrt(mean_squared_error(a,p)))
def MSE(a,p): return float(mean_squared_error(a,p))
def MAPE(a,p):
    a=np.array(a,dtype=float); p=np.array(p,dtype=float)
    a[a==0]=1e-8
    return float(np.mean(np.abs((a-p)/a))*100)


# ------------------------------------------------------------
# SYMBOL RESOLVER (NEW)
# ------------------------------------------------------------
def resolve_symbol(file_name: str) -> str:
    """
    Convert uploaded CSV name -> proper NSE/TradingView symbol.
    Example: Hdfc.csv → HDFCBANK.NS
    """
    base = os.path.splitext(file_name)[0].upper()
    base = base.replace(" ", "").replace("-", "").replace("_", "")

    # Mappings for Indian stocks
    nse_map = {
        "HDFC": "HDFCBANK.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "WIPRO": "WIPRO.NS",
        "SBIN": "SBIN.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "ASIANPAINT": "ASIANPAINT.NS",
        "MARUTI": "MARUTI.NS",
        "KOTAKBANK": "KOTAKBANK.NS",
        "AXISBANK": "AXISBANK.NS",
        "HCLTECH": "HCLTECH.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS",
        "ADANIPORTS": "ADANIPORTS.NS",
        "ADANIPOWER": "ADANIPOWER.NS",
        "ADANIENT": "ADANIENT.NS",
    }

    if base in nse_map:
        return nse_map[base]

    if base.endswith(".NS"):
        return base

    # Default NSE assumption
    return base + ".NS"


# ------------------------------------------------------------
# DETECT DATE & PRICE COLUMN
# ------------------------------------------------------------
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower(): return c
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close","close","Adj Close","price","Price"]:
        if c in df.columns:
            return c
    numeric=df.select_dtypes(include="number").columns
    if len(numeric)==0:
        raise ValueError("No numeric column found")
    return numeric[-1]


# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------
def plot_series_buf(train,test,pred,title):
    fig,ax=plt.subplots(figsize=(10,4))
    train.plot(ax=ax,label="Train")
    test.plot(ax=ax,label="Test")
    pred.plot(ax=ax,label="Forecast")
    ax.set_title(title)
    ax.legend()
    buf=io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def combined_chart_buf(train,test,preds):
    fig,ax=plt.subplots(figsize=(12,5))
    train.plot(ax=ax,label="Train",linewidth=2)
    test.plot(ax=ax,label="Test",linewidth=2)
    for name,p in preds.items():
        ax.plot(p.index,p.values,label=name,linewidth=2)
    ax.legend()
    ax.set_title("Combined Forecasts")
    buf=io.BytesIO()
    fig.savefig(buf,format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def radar_chart_buf(df):
    df=df[["RMSE","MSE","MAPE"]].astype(float)
    norm=(df.max()-df)/(df.max()-df.min()+1e-8)
    labels=list(norm.columns)
    angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
    angles+=angles[:1]
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111,polar=True)
    for idx in norm.index:
        vals=list(norm.loc[idx])+[norm.loc[idx][0]]
        ax.plot(angles,vals,label=idx)
        ax.fill(angles,vals,alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    buf=io.BytesIO()
    fig.savefig(buf,format="png")
    buf.seek(0)
    plt.close(fig)
    return buf



# ------------------------------------------------------------
# TRADINGVIEW DASHBOARD (UPDATED to use manual override)
# ------------------------------------------------------------
def tradingview_dashboard(symbol: str):

    # TICKER TAPE
    tape = """
    <div class="tradingview-widget-container">
    <script src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js">
    {
      "symbols":[
        {"proName":"NSE:NIFTY","title":"NIFTY 50"},
        {"proName":"BSE:SENSEX","title":"SENSEX"},
        {"proName":"CRYPTO:BTCUSD","title":"BTC"},
        {"proName":"CRYPTO:ETHUSD","title":"ETH"}
      ],
      "colorTheme":"dark",
      "displayMode":"adaptive"
    }
    </script></div>
    """
    components.html(tape, height=80)

    # MAIN CHART
    chart=f"""
    <div class="tradingview-widget-container">
      <div id="tv_chart"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "width":"100%",
        "height":550,
        "symbol":"{symbol}",
        "interval":"D",
        "timezone":"Etc/UTC",
        "theme":"dark",
        "style":"1",
        "locale":"en",
        "toolbar_bg":"#000000",
        "allow_symbol_change":true,
        "container_id":"tv_chart"
      }});
      </script>
    </div>
    """
    components.html(chart, height=560)

    # SCREENER
    screener = """
    <div class="tradingview-widget-container">
      <script src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js">
      {
        "width":"100%",
        "height":"550",
        "defaultColumn":"overview",
        "defaultScreen":"most_capitalized",
        "market":"india",
        "showToolbar":true,
        "colorTheme":"dark"
      }
      </script>
    </div>
    """
    components.html(screener, height=560)




# ==================================================================
# MAIN APP
# ==================================================================
def main():
    st.title("📈 Premium Forecast Dashboard (Updated Symbol Handling)")

    # -----------------------------  
    # UPLOAD  
    # -----------------------------
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.stop()

    # Read file
    df = pd.read_csv(uploaded)
    df.columns=[c.strip() for c in df.columns]

    # Detect columns
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)

    price_col = detect_price_column(df)
    series = df[price_col].astype(float)

    # Preview
    with st.expander("📌 Data Preview", True):
        st.dataframe(df.tail())
        fig, ax = plt.subplots(figsize=(10,3))
        series.plot(ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------  
    # SYMBOL AUTO + MANUAL OVERRIDE  
    # -----------------------------
    auto_symbol = resolve_symbol(uploaded.name)

    st.sidebar.subheader("TradingView Symbol Override")
    tv_symbol = st.sidebar.text_input(
        "Symbol (Auto detected from file):",
        value=auto_symbol,
        help="e.g., HDFCBANK.NS, TCS.NS, RELIANCE.NS, AAPL, BTCUSD"
    ).upper().strip()

    # SHOW TV DASHBOARD
    st.subheader("📊 TradingView Market Dashboard")
    tradingview_dashboard(tv_symbol)


    # ------------------------------------------------------------
    # PREP DATA FOR MODELS
    # ------------------------------------------------------------
    series_daily = prepare_series(df, col=price_col, freq="D")
    train, test = train_test_split_series(series_daily, test_size=0.2)

    # -----------------------------
    # SIDEBAR MODEL SETTINGS
    # -----------------------------
    st.sidebar.subheader("Forecast Models")
    options = ["ARIMA","SARIMA","LSTM"]
    if PROPHET_AVAILABLE:
        options.append("Prophet")

    selected = st.sidebar.multiselect("Select models", options, default=options)

    # ARIMA settings
    arima_order = tuple(map(int, st.sidebar.text_input("ARIMA (p,d,q)", "5,1,0").split(",")))

    # SARIMA settings
    sarima_order = tuple(map(int, st.sidebar.text_input("SARIMA (p,d,q)", "1,1,1").split(",")))
    seasonal_order = tuple(map(int, st.sidebar.text_input("Seasonal (P,D,Q,s)", "1,1,1,12").split(",")))

    # LSTM settings
    lstm_seq = st.sidebar.number_input("LSTM sequence length", 10, 300, 60)
    lstm_epochs = st.sidebar.number_input("LSTM epochs", 1, 50, 5)
    lstm_batch = st.sidebar.number_input("LSTM batch size", 1, 256, 32)

    run = st.sidebar.button("🚀 Run Models")

    combined = {}
    scores = {}

    col1, col2 = st.columns(2)

    if run:

        # ---------------- ARIMA ----------------
        if "ARIMA" in selected:
            with st.spinner("Running ARIMA..."):
                try:
                    model = train_arima(train.squeeze(), order=arima_order)
                    pred_vals = forecast_arima(model, steps=len(test))
                    pred = pd.Series(pred_vals, index=test.index)

                    combined["ARIMA"] = pred
                    scores["ARIMA"] = {
                        "RMSE": RMSE(test,pred),
                        "MSE": MSE(test,pred),
                        "MAPE": MAPE(test,pred)
                    }

                    col1.subheader("ARIMA Forecast")
                    col1.image(plot_series_buf(train,test,pred,"ARIMA"))
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

        # ---------------- SARIMA ----------------
        if "SARIMA" in selected:
            with st.spinner("Running SARIMA..."):
                try:
                    model = train_sarima(train.squeeze(), order=sarima_order, seasonal_order=seasonal_order)
                    pred_vals = forecast_sarima(model, steps=len(test))
                    pred = pd.Series(pred_vals, index=test.index)

                    combined["SARIMA"] = pred
                    scores["SARIMA"] = {
                        "RMSE": RMSE(test,pred),
                        "MSE": MSE(test,pred),
                        "MAPE": MAPE(test,pred)
                    }

                    col1.subheader("SARIMA Forecast")
                    col1.image(plot_series_buf(train,test,pred,"SARIMA"))
                except Exception as e:
                    st.error(f"SARIMA failed: {e}")

        # ---------------- PROPHET ----------------
        if "Prophet" in selected and PROPHET_AVAILABLE:
            with st.spinner("Running Prophet..."):
                try:
                    model = train_prophet(train.squeeze())
                    pred_vals = forecast_prophet(model, periods=len(test))
                    pred_vals = pred_vals.reindex(test.index)
                    pred = pd.Series(pred_vals.values, index=test.index)

                    combined["Prophet"] = pred
                    scores["Prophet"] = {
                        "RMSE": RMSE(test,pred),
                        "MSE": MSE(test,pred),
                        "MAPE": MAPE(test,pred)
                    }

                    col2.subheader("Prophet Forecast")
                    col2.image(plot_series_buf(train,test,pred,"Prophet"))
                except Exception as e:
                    st.error(f"Prophet failed: {e}")

        # ---------------- LSTM ----------------
        if "LSTM" in selected:
            with st.spinner("Running LSTM..."):
                try:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(series_daily.values.reshape(-1,1))
                    split = int(len(scaled)*0.8)
                    train_scaled = scaled[:split]

                    model = train_lstm(train_scaled, seq_len=lstm_seq, epochs=lstm_epochs, batch_size=lstm_batch)
                    pred_vals = forecast_lstm(model, scaled, scaler, seq_len=lstm_seq, steps=len(test))
                    pred = pd.Series(pred_vals, index=test.index)

                    combined["LSTM"] = pred
                    scores["LSTM"] = {
                        "RMSE": RMSE(test,pred),
                        "MSE": MSE(test,pred),
                        "MAPE": MAPE(test,pred)
                    }

                    col2.subheader("LSTM Forecast")
                    col2.image(plot_series_buf(train,test,pred,"LSTM"))
                except Exception as e:
                    st.error(f"LSTM failed: {e}")


        # ------------------------------------------------------------
        # COMBINED CHART
        # ------------------------------------------------------------
        if combined:
            st.subheader("📊 Combined Forecast Chart")
            buf = combined_chart_buf(train, test, combined)
            st.image(buf)
            st.download_button("Download Combined Chart", buf.getvalue(), "combined.png", "image/png")

        # ------------------------------------------------------------
        # METRICS
        # ------------------------------------------------------------
        if scores:
            st.subheader("📈 Model Performance Metrics")

            dfm = pd.DataFrame(scores).T
            dfm = dfm.sort_values("RMSE")
            dfm["Rank"] = range(1, len(dfm)+1)

            st.dataframe(dfm)

            st.success(f"🏆 Best Model: {dfm.index[0]}")

            metrics_csv = dfm.to_csv().encode("utf-8")
            st.download_button("Download metrics CSV", metrics_csv, "metrics.csv")

            radar = radar_chart_buf(dfm)
            st.subheader("📡 Radar Chart")
            st.image(radar)
            st.download_button("Download Radar Chart", radar.getvalue(), "radar.png", "image/png")




if __name__ == "__main__":
    main()
