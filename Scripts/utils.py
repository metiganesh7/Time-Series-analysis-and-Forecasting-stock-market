
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def download_data(ticker='^NSEI', start='2015-01-01', end=None, save_path='data'):
    os.makedirs(save_path, exist_ok=True)
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(os.path.join(save_path, f"{ticker.replace('/','_')}.csv"))
    return df

def load_csv(path):
    return pd.read_csv(path, parse_dates=['Date'], index_col='Date')

def prepare_series(df, col='Close', freq='D'):
    s = df[[col]].resample(freq).last().ffill()
    return s

def train_test_split_series(series, test_size=0.2):
    n = len(series)
    split = int(n*(1-test_size))
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test

def scale_series(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train.reshape(-1,1) if isinstance(train, np.ndarray) else train.values.reshape(-1,1))
    train_s = scaler.transform(train.values.reshape(-1,1))
    test_s = scaler.transform(test.values.reshape(-1,1))
    return train_s, test_s, scaler
