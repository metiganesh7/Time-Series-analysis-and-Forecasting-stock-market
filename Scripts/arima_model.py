
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima(train, order=(5,1,0), save_path='models', name='arima_model.pkl'):
    os.makedirs(save_path, exist_ok=True)
    model = ARIMA(train, order=order)
    res = model.fit()
    joblib.dump(res, os.path.join(save_path, name))
    return res

def forecast_arima(model_res, steps=30):
    pred = model_res.forecast(steps=steps)
    return pred
