import warnings
warnings.filterwarnings('ignore')

import joblib
import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_arima(train_series, order=(5,1,0), save_path='models', name='arima_model.pkl'):
    os.makedirs(save_path, exist_ok=True)

    # Ensure 1D series
    if isinstance(train_series, pd.DataFrame):
        train_series = train_series.squeeze()

    model = ARIMA(train_series, order=order)
    result = model.fit()

    joblib.dump(result, os.path.join(save_path, name))
    return result


def forecast_arima(model, steps=30):
    forecast = model.forecast(steps=steps)
    return forecast.tolist()
