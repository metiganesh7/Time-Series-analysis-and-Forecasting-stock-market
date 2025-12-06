import warnings
warnings.filterwarnings('ignore')

import joblib
import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                 save_path='models', name='sarima_model.pkl'):
    """
    Train a SARIMA model on a pandas Series or DataFrame column.
    """

    os.makedirs(save_path, exist_ok=True)

    # If train is a DataFrame, convert to Series
    if isinstance(train, pd.DataFrame):
        train = train.iloc[:, 0]

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    res = model.fit(disp=False)

    joblib.dump(res, os.path.join(save_path, name))
    return res


def forecast_sarima(model_res, steps=30):
    """
    Forecast next N time steps using a fitted SARIMA model.
    """
    pred = model_res.get_forecast(steps=steps).predicted_mean
    return pred
