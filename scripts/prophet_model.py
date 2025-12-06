from prophet import Prophet
import pandas as pd
import os
import joblib

def train_prophet(train_series, save_path='models', name='prophet_model.pkl'):
    os.makedirs(save_path, exist_ok=True)

    if isinstance(train_series, pd.DataFrame):
        train_series = train_series.squeeze()

    df = pd.DataFrame({
        "ds": pd.to_datetime(train_series.index),
        "y": train_series.values.reshape(-1)
    })

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    joblib.dump(model, os.path.join(save_path, name))
    return model


def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"][-periods:]
