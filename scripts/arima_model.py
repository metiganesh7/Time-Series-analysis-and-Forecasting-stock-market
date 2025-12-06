import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

import joblib
import os

def train_arima(train, order=(5,1,0), save_path='models', name='arima_model.pkl'):
    """
    Train an ARIMA model and save it to disk.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # If train is a DataFrame, convert to Series
    if isinstance(train, pd.DataFrame):
        train = train.iloc[:,0]

    model = ARIMA(train, order=order)
    res = model.fit()

    joblib.dump(res, os.path.join(save_path, name))
    return res


def forecast_arima(model_res, steps=30):
    """
    Forecast future values using a trained ARIMA model.
    """
    pred = model_res.forecast(steps=steps)
    return pred
