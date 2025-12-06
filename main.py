import sys
import os
import io
import datetime as dt

# Ensure scripts import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

# Model imports
from scripts.utils import prepare_series, train_test_split_series
from scripts.arima_model import train_arima, forecast_arima
from scripts.sarima_model import train_sarima, forecast_sarima
from scripts.lstm_model import train_lstm, forecast_lstm

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit.components.v1 as components

# --------------------------
# SAFE PROPHET IMPORT
# --------------------------
try:
    from scripts.prophet_model import train_prophet, forecast_prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False


# --------------------------
# PREMIUM UI THEME (CSS)
# --------------------------
st.markdown("""
<style>

html, body, .stApp {
    background-color: #0A0F1F !important;
    font-family: 'Poppins', sans-serif;
    color: #E4E8F0;
}

.block-container {
    padding: 2rem 3rem;
    background: rgba(255,255,255,0.04);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
}

.sidebar .sidebar-content {
    background: rgba(255,255,255,0.07) !important;
    padding-top: 2rem;
    border-right: 1px solid rgba(255,255,255,0.05);
}

h1, h2, h3 {
    color: #E9EDFA !important;
    font-weight: 600;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #6B73FF, #000DFF);
    color: white;
    padding: 12px 22px;
    border-radius: 12px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    transition: 0.25s;
}
.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,255,0.5);
}

/* Download buttons */
.stDownloadButton button {
    background: linear-gradient(135deg, #FFB300, #FFDD55);
    color: black !important;
    padding: 10px 18px;
    border-radius: 12px;
    font-weight: 700;
    border: none;
    transition: 0.25s;
}
.stDownloadButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(255,200,0,0.5);
}

.chart-card {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 18px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# METRIC FUNCTIONS
# --------------------------
def RMSE(a,p): return np.sqrt(mean_squared_error(a,p))
def MSE(a,p): return mean_squared_error(a,p)
def MAPE(a,p):
    a=np.array(a); p=np.array(p)
    a[a==0]=1e-8
    return np.mean(np.abs((a-p)/a))*100

# --------------------------
# HELPERS
# --------------------------
def detect_date_column(df):
    for c in df.columns:
        if "date" in c.lower(): return c
    return df.columns[0]

def detect_price_column(df):
    for c in ["Close","close","Adj Close","Price","price"]:
        if c in df.columns: return c
    return df.select_dtypes("number").columns[-1]

def plot_series_buf(train,test,pred,title):
    fig, ax = plt.subplots(figsize=(10,4))
    train.plot(ax=ax,label="Train")
    test.plot(ax=ax,label="Test")
    pred.plot(ax=ax,label="Forecast")
    ax.legend(); ax.set_title(title)
    buf=io.BytesIO(); fig.savefig(buf,format="png"); buf.seek(0)
    plt.close(fig)
    return buf

def plot_combined_chart(train,test,preds):
    fig, ax = plt.subplots(figsize=(12,5))
    train.plot(ax=ax,label="Train",linewidth=2)
    test.plot(ax=ax,label="Test",linewidth=2)
    for name,fc in preds.items():
        ax.plot(fc.index,fc.values,label=name,linewidth=2)
    ax.legend(); ax.set_title("Combined Forecasts")
    buf=io.BytesIO(); fig.savefig(buf,format="png"); buf.seek(0)
    plt.close(fig)
    return buf

def create_radar(df):
    df=df[["RMSE","MSE","MAPE"]].astype(float)
    norm=(df.max()-df)/(df.max()-df.min()+1e-8)
    labels=list(norm.columns)
    angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()+[0]
    fig=plt.figure(figsize=(6,6)); ax=fig.add_subplot(111,polar=True)
    for idx in norm.index:
        vals=list(norm.loc[idx])+[norm.loc[idx][0]]
        ax.plot(angles,vals,label=idx,linewidth=2)
        ax.fill(angles,vals,alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]),labels)
    buf=io.BytesIO()
    fig.savefig(buf,format="png"); buf.seek(0)
    plt.close(fig)
    return buf

# --------------------------
# UI
# --------------------------
st.title("📈 Premium Stock Forecasting Dashboard")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if not file: st.stop()

df = pd.read_csv(file)
df.columns=[c.strip() for c in df.columns]
date_col = detect_date_column(df)
df[date_col]=pd.to_datetime(df[date_col])
df=df.set_index(date_col)

price_col = detect_price_column(df)
series = df[price_col]

# Preview
with st.expander("📋 Data Preview", True):
    st.dataframe(df.tail())
    fig, ax = plt.subplots(figsize=(10,3))
    series.plot(ax=ax); st.pyplot(fig); plt.close(fig)

series = prepare_series(df,col=price_col,freq="D")
train,test = train_test_split_series(series,0.2)

# --------------------------
# TRADINGVIEW DASHBOARD
# --------------------------
st.subheader("📊 TradingView Market Dashboard")

file_name = os.path.splitext(file.name)[0]
tv_symbol = file_name.upper() if "." in file_name else file_name.upper()+".NS"

# 1️⃣ TICKER TAPE
ticker = """
<div class="tradingview-widget-container">
<script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js">
{
"symbols":[
{"proName":"NSE:NIFTY","title":"NIFTY 50"},
{"proName":"BSE:SENSEX","title":"SENSEX"},
{"proName":"CRYPTO:BTCUSD","title":"Bitcoin"},
{"proName":"CRYPTO:ETHUSD","title":"Ethereum"}
],
"colorTheme":"dark",
"displayMode":"adaptive"
}
</script>
</div>
"""
components.html(ticker,height=80)

# 2️⃣ ADVANCED CHART
chart = f"""
<div class="tradingview-widget-container">
<div id="tradingview_chart"></div>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
new TradingView.widget({{
"width":"100%",
"height":550,
"symbol":"{tv_symbol}",
"interval":"D",
"timezone":"Etc/UTC",
"theme":"dark",
"style":"1",
"locale":"en",
"toolbar_bg":"#000000",
"hide_top_toolbar":false,
"allow_symbol_change":true,
"container_id":"tradingview_chart"
}});
</script>
</div>
"""
components.html(chart,height=560)

# 3️⃣ MARKET OVERVIEW
market = """
<div class="tradingview-widget-container">
<script src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js">
{
"colorTheme":"dark",
"dateRange":"12M",
"showChart":true,
"locale":"en",
"height":"500",
"tabs":[
{"title":"Indices","symbols":[
{"s":"NSE:NIFTY"},
{"s":"BSE:SENSEX"},
{"s":"NASDAQ:NDX"}
]}
]}
</script>
</div>
"""
components.html(market,height=520)

# 4️⃣ TECHNICAL ANALYSIS
ta = f"""
<div class="tradingview-widget-container">
<script src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js">
{{
"symbol":"{tv_symbol}",
"interval":"1D",
"height":400,
"width":"100%",
"colorTheme":"dark"
}}
</script>
</div>
"""
components.html(ta,height=420)

# 5️⃣ STOCK SCREENER
screener = """
<div class="tradingview-widget-container">
<script src="https://s3.tradingview.com/external-embedding/embed-widget-screener.js">
{
"width":"100%",
"height":600,
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
components.html(screener,height=620)

# --------------------------
# MODEL CONFIG
# --------------------------
st.sidebar.header("Models")

opts=["ARIMA","SARIMA","LSTM"]
if PROPHET_AVAILABLE: opts.append("Prophet")

models=st.sidebar.multiselect("Select Models",opts,default=opts)

def parse(text,n):
    try: return tuple([int(x) for x in text.split(",")][:n])
    except: return (1,1,1)[:n]

arima_order=parse(st.sidebar.text_input("ARIMA (p,d,q)","5,1,0"),3)
sarima_order=parse(st.sidebar.text_input("SARIMA (p,d,q)","1,1,1"),3)
seasonal_order=parse(st.sidebar.text_input("Seasonal (P,D,Q,s)","1,1,1,12"),4)

lstm_seq=st.sidebar.number_input("LSTM seq len",10,200,60)
lstm_ep=st.sidebar.number_input("LSTM epochs",1,50,5)
lstm_bs=st.sidebar.number_input("Batch size",1,256,32)

run=st.sidebar.button("🚀 Run Models")

combined={}
scores={}

# --------------------------
# RUN MODELS
# --------------------------
if run:

    # ARIMA
    if "ARIMA" in models:
        with st.spinner("Running ARIMA..."):
            m=train_arima(train.squeeze(),order=arima_order)
            pred=pd.Series(forecast_arima(m,len(test)),index=test.index)
            combined["ARIMA"]=pred
            scores["ARIMA"]={"RMSE":RMSE(test,pred),"MSE":MSE(test,pred),"MAPE":MAPE(test,pred)}
            st.image(plot_series_buf(train,test,pred,"ARIMA Forecast"))

    # SARIMA
    if "SARIMA" in models:
        with st.spinner("Running SARIMA..."):
            m=train_sarima(train.squeeze(),order=sarima_order,seasonal_order=seasonal_order)
            pred=pd.Series(forecast_sarima(m,len(test)),index=test.index)
            combined["SARIMA"]=pred
            scores["SARIMA"]={"RMSE":RMSE(test,pred),"MSE":MSE(test,pred),"MAPE":MAPE(test,pred)}
            st.image(plot_series_buf(train,test,pred,"SARIMA Forecast"))

    # Prophet
    if "Prophet" in models and PROPHET_AVAILABLE:
        with st.spinner("Running Prophet..."):
            try:
                m=train_prophet(train.squeeze())
                fc=forecast_prophet(m,len(test)).reindex(test.index)
                pred=pd.Series(fc.values,index=test.index)
                combined["Prophet"]=pred
                scores["Prophet"]={"RMSE":RMSE(test,pred),"MSE":MSE(test,pred),"MAPE":MAPE(test,pred)}
                st.image(plot_series_buf(train,test,pred,"Prophet Forecast"))
            except Exception as e:
                st.error(f"Prophet failed: {e}")

    # LSTM
    if "LSTM" in models:
        with st.spinner("Running LSTM..."):
            sc=MinMaxScaler()
            scaled=sc.fit_transform(series.values.reshape(-1,1))
            split=int(len(scaled)*0.8)
            lstm_data=scaled[:split]
            lstm=train_lstm(lstm_data,seq_len=lstm_seq,epochs=lstm_ep,batch_size=lstm_bs)
            fc=forecast_lstm(lstm,scaled,sc,seq_len=lstm_seq,steps=len(test))
            pred=pd.Series(fc,index=test.index)
            combined["LSTM"]=pred
            scores["LSTM"]={"RMSE":RMSE(test,pred),"MSE":MSE(test,pred),"MAPE":MAPE(test,pred)}
            st.image(plot_series_buf(train,test,pred,"LSTM Forecast"))

# --------------------------
# RESULTS
# --------------------------
if combined:
    st.subheader("📌 Combined Forecast Chart")
    buf=plot_combined_chart(train,test,combined)
    st.image(buf)
    st.download_button("Download Combined Chart",buf.getvalue(),"combined.png","image/png")

if scores:
    st.subheader("📊 Metrics & Model Ranking")
    dfm=pd.DataFrame(scores).T.sort_values("RMSE")
    dfm["Rank"]=range(1,len(dfm)+1)
    st.dataframe(dfm)
    st.success(f"🏆 Best Model: {dfm.index[0]}")

    radar=create_radar(dfm)
    st.subheader("📡 Radar Chart")
    st.image(radar)
    st.download_button("Download Radar Chart",radar.getvalue(),"radar.png","image/png")
