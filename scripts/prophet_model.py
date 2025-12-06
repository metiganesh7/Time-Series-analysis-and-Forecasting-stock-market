
from prophet import Prophet
import pandas as pd
import os
import joblib

def train_prophet(train_series, save_path='models', name='prophet_model.pkl'):
    os.makedirs(save_path, exist_ok=True)
    df = train_series.reset_index().rename(columns={'Date':'ds', train_series.name:'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    joblib.dump(m, os.path.join(save_path, name))
    return m

def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds','yhat']].set_index('ds')['yhat'][-periods:]
