import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib


def create_sequences(data, seq_len=60):
    """
    Create LSTM training sequences.
    data shape must be (n, 1)
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape to (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def build_lstm(input_shape):
    """
    Build 2-layer LSTM forecasting model.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(train_scaled, seq_len=60, epochs=10, batch_size=32,
               save_path='models', name='lstm_model.h5'):
    """
    Train LSTM on scaled training data.
    """
    os.makedirs(save_path, exist_ok=True)

    X_train, y_train = create_sequences(train_scaled, seq_len=seq_len)

    model = build_lstm((X_train.shape[1], 1))

    es = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    model.save(os.path.join(save_path, name))

    return model


def forecast_lstm(model, full_series_scaled, scaler, seq_len=60, steps=30):
    """
    Forecast future values using a trained LSTM model.
    full_series_scaled shape must be (n, 1)
    """
    last_seq = full_series_scaled[-seq_len:].copy()
    preds = []

    for _ in range(steps):
        X = last_seq.reshape((1, seq_len, 1))
        pred = model.predict(X, verbose=0)[0, 0]
        preds.append(pred)

        # update sequence
        last_seq = np.append(last_seq[1:], pred)

    preds = np.array(preds).reshape(-1, 1)

    # invert scaling
    inv = scaler.inverse_transform(preds)

    return inv.flatten()
