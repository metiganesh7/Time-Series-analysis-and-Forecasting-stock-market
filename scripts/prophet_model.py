from prophet import Prophet
import pandas as pd
import os
import joblib

def train_prophet(train_series, save_path='models', name='prophet_model.pkl'):
    """
    Train a Prophet forecasting model.
    Ensures that input is a clean 1D pandas Series.
    """
    os.makedirs(save_path, exist_ok=True)

    # Ensure Prophet gets a clean 1D Series
    if isinstance(train_series, pd.DataFrame):
        train_series = train_series.squeeze()

    # Force y to be 1-D and ds to be datetime
    df = pd.DataFrame({
        "ds": pd.to_datetime(train_series.index),
        "y": train_series.values.reshape(-1)  # flatten to 1-D
    })

    m = Prophet(daily_seasonality=True)
    m.fit(df)

    joblib.dump(m, os.path.join(save_path, name))
    return m


def forecast_prophet(model, periods=30):
    """
    Forecast future values using Prophet model.
    Returns only the last N periods aligned with test index.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # return only final forecast values
    yhat = forecast.set_index("ds")["yhat"]
    return yhat[-periods:]
