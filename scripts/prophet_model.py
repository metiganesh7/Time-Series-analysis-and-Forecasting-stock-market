from prophet import Prophet
import pandas as pd
import os
import joblib


def train_prophet(train_series, save_path='models', name='prophet_model.pkl'):
    """
    Train a Prophet model using a pandas Series with a DateTimeIndex.
    """
    os.makedirs(save_path, exist_ok=True)

    # Convert series to Prophet-friendly DataFrame
    df = pd.DataFrame({
        'ds': train_series.index,   # Prophet date column
        'y': train_series.values    # target variable
    })

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    joblib.dump(model, os.path.join(save_path, name))
    return model


def forecast_prophet(model, periods=30):
    """
    Forecast future values for N periods using Prophet.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Extract last N predictions
    yhat = forecast[['ds', 'yhat']].set_index('ds')['yhat'].iloc[-periods:]

    return yhat
