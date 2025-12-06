import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ===============================
# ✅ PREPARE TIME SERIES
# ===============================
def prepare_series(df, col, freq="D"):
    series = df[col].astype(float)
    series.index = pd.to_datetime(series.index)
    series = series.asfreq(freq)
    series = series.fillna(method="ffill")
    return series

# ===============================
# ✅ TRAIN / TEST SPLIT
# ===============================
def train_test_split_series(series, test_size=0.2):
    split = int(len(series) * (1 - test_size))
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test

# ===============================
# ✅ MINMAX SCALING (FOR LSTM)
# ===============================
def scale_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    return scaled, scaler
