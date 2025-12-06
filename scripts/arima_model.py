import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings('ignore')
import joblib, os


def train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12), save_path='models', name='sarima_model.pkl'):
    os.makedirs(save_path, exist_ok=True)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    joblib.dump(res, os.path.join(save_path, name))
    return res

def forecast_sarima(model_res, steps=30):
    pred = model_res.get_forecast(steps=steps).predicted_mean
    return pred
