import warnings
warnings.filterwarnings('ignore')

import joblib
import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima(train_series, order=(1,1,1), seasonal_order=(1,1,1,12),
                 save_path='models', name='sarima_model.pkl'):
    os.makedirs(save_path, exist_ok=True)

    if isinstance(train_series, pd.DataFrame):
        train_series = train_series.squeeze()

    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)

    joblib.dump(result, os.path.join(save_path, name))
    return result


def forecast_sarima(model, steps=30):
    pred = model.get_forecast(steps=steps).predicted_mean
    return pred.tolist()
