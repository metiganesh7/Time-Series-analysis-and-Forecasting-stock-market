# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('Adani stock time series (1).csv')

# %%
df.info()

# %%
df.head()

# %%
df['Close'].dtypes

# %%
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Close'].isna().sum()

# %%
df['Log_Close'] = np.log(df['Close'])

# %%
df['Log_Close_Diff'] = df['Log_Close'].diff()

# %%
df[['Date', 'Close', 'Log_Close', 'Log_Close_Diff']].head()

# %%
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.title('Adani Ports Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# %%
# 20-day and 50-day moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA20'], label='20-Day MA', linestyle='--')
plt.plot(df['MA50'], label='50-Day MA', linestyle='--')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# %%
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows where Date conversion failed
df = df.dropna(subset=['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Now resample monthly
monthly = df['Close'].resample('ME').mean()


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(monthly, marker='o', label='Monthly Avg Close')
plt.title('Monthly Stock Price Trend (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Average Close Price')
plt.legend()
plt.show()


# %%
yearly = df['Close'].resample('Y').mean()

plt.figure(figsize=(10,5))
plt.plot(yearly, marker='o', label='Yearly Avg Close')
plt.title('Yearly Stock Price Pattern')
plt.xlabel('Year')
plt.ylabel('Average Close Price')
plt.legend()
plt.show()


# %%
plt.figure(figsize=(14,6))
plt.plot(df['Log_Close_Diff'], label='Log Returns')
plt.title('Log Returns of Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()


# %%
#ARIMA MODEL

# %%
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# %%
df['Close'] = df['Close'].astype(float)
ts = df['Close']

# %%
result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    print("Time series is non-stationary. Differencing needed.")
else:
    print("Time series is stationary. ARIMA can be applied directly.")

# %%
ts_diff = ts.diff().dropna()

# %%
plt.figure(figsize=(12,5))
plot_acf(ts_diff, lags=40)
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(ts_diff, lags=40)
plt.show()

# %%
df = df.asfreq('B')  # 'B' = business day frequency


# %%
# Example: ARIMA(1,1,1) → p=1, d=1, q=1
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# %%
forecast = model_fit.get_forecast(steps=30)
forecast_index = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=30, freq='B')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1],
                 color='pink', alpha=0.3)
plt.title('ARIMA Forecast for Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
#SARIMA MODEL

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# %%
# Ensure Date column exists and is datetime
df['Date'] = pd.to_datetime(df.index) if 'Date' not in df.columns else pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Make sure the index has a frequency
df = df.asfreq('B')  # 'B' = Business day frequency

# Use Close price
ts = df['Close']


# %%
# If you are using Close price directly
ts = df['Close'].copy()

# Drop missing values
ts = ts.dropna()

# OR fill missing values (optional)
# ts = ts.fillna(method='ffill')  # forward fill
# ts = ts.fillna(method='bfill')  # backward fill

# Now run ADF test
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# %%
result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, series is non-stationary → differencing needed


# %%
import pmdarima as pm

# Automatically find best SARIMA parameters
smodel = pm.auto_arima(ts,
                       seasonal=True,
                       m=5,      # weekly seasonality (adjust based on data)
                       stepwise=True,
                       trace=True,
                       suppress_warnings=True)
print(smodel.summary())


# %%
model = SARIMAX(ts,
                order=(1,1,1),
                seasonal_order=(1,1,1,5),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit()
print(model_fit.summary())


# %%
forecast_steps = 30  # Next 30 business days
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=ts.index[-1]+pd.Timedelta(days=1),
                               periods=forecast_steps, freq='B')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(ts, label='Actual')
plt.plot(forecast_index, forecast_values, color='red', label='Forecast')
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1],
                 color='pink', alpha=0.3)
plt.title('SARIMA Forecast for Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
#PROPHET MODEL

# %%
pip install prophet

# %%
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# %%
# Ensure Date column exists and is datetime
df['Date'] = pd.to_datetime(df.index) if 'Date' not in df.columns else pd.to_datetime(df['Date'])

# Create Prophet DataFrame
prophet_df = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})

# Handle missing values
prophet_df = prophet_df.dropna()


# %%
m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)

# Fit the model
m.fit(prophet_df)


# %%
# Forecast next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# View forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# %%
# Plot forecast
fig1 = m.plot(forecast)
plt.title('Prophet Forecast for Adani Ports')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Plot components: trend, weekly, yearly seasonality
fig2 = m.plot_components(forecast)
plt.show()


# %%
#LSTM

# %%
!pip install tensorflow

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# %%
# Use Close price
data = df[['Close']].values

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)


# %%
X = []
y = []
time_steps = 60  # number of past days to look at

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape X for LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# %%
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')



# %%
model.summary()


# %%
history = model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, verbose=1)


# %%
# Prepare test data (use last 60 days)
last_60_days = scaled_data[-time_steps:]
X_test = np.array([last_60_days])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print("Predicted Next Close Price:", predicted_price[0][0])


# %%
plt.figure(figsize=(12,6))
plt.plot(df.index[-len(y):], scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Price', color='blue')
plt.plot(df.index[-len(y):], scaler.inverse_transform(model.predict(X)), label='LSTM Predicted', color='red')
plt.title('Adani Ports Stock Price Prediction (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
print("Model built and compiled successfully ✅")

# %%
#MODEL EVALUATION

# %%
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]  # include last 60 points for continuity

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# %%
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)


# %%
X_test = []
y_test = scaled_data[train_size:, 0]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# %%
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # scale back to original values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# %%
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# %%
predictions = scaler.inverse_transform(predictions)  # should already be 2D


# %%
# Check shapes
print("y_test_actual shape:", y_test_actual.shape)
print("predictions shape:", predictions.shape)

# Make them equal
min_len = min(len(y_test_actual), len(predictions))
y_test_actual = y_test_actual[:min_len]
predictions = predictions[:min_len]


# %%
mask = ~np.isnan(y_test_actual.flatten()) & ~np.isnan(predictions.flatten())
y_test_actual = y_test_actual[mask]
predictions = predictions[mask]


# %%
time_steps = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]  # include last 60 points from train

# Create X_test and y_test
X_test, y_test = [], []
for i in range(time_steps, len(test_data)):
    X_test.append(test_data[i-time_steps:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Check lengths
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Reshape for LSTM
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# %%
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))


# %%
# -------------------------------
# LSTM for Adani Ports Stock Price
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1️⃣ Prepare Data
df = pd.read_csv('Adani stock time series (1).csv')  # replace with your file path
df['Date'] = pd.to_datetime(df['Date'])  # ensure date is datetime
df = df.sort_values('Date')
data = df[['Close']].values  # using Close price

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# 2️⃣ Train-Test Split
time_steps = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]  # include last 60 points from train

# Create sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# 3️⃣ Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()  # view model architecture

# 4️⃣ Train Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 5️⃣ Predict & Inverse Scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# 6️⃣ Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
mape = np.mean(np.abs((y_test_actual - predictions)/y_test_actual)) * 100

print("\n📊 Model Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# 7️⃣ Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Adani Ports Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# 8️⃣ Plot Training Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# %%
#FORECASTING & FUTURE PREDICTION

# %%
# -------------------------------
# Forecast Next 30 Days
# -------------------------------

future_days = 30
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)
    
    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='30-Day Forecast', marker='o')
plt.title('Adani Ports Stock Price 30-Day Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
#PREDICTING FUTURE FOR NEXT 90 DAYS

# %%
# -------------------------------
# Forecast Next 90 Days
# -------------------------------

future_days = 90
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)
    
    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

# Plot forecast
plt.figure(figsize=(14,7))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='90-Day Forecast', marker='o')
plt.title('Adani Ports Stock Price 90-Day Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
#PREDICTING FUTURE FOR NEXT 6 MONTHS

# %%
# -------------------------------
# Forecast Next 6 Months (~132 Trading Days)
# -------------------------------

future_days = 132  # approx 6 months
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)
    
    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=future_days)  # business days

# Plot forecast
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='6-Month Forecast', marker='o', markersize=3)
plt.title('Adani Ports Stock Price 6-Month Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
# Convert future_dates to numpy array
future_dates_array = future_dates.to_numpy()

plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price', color='blue')
plt.plot(future_dates_array, forecast, label='6-Month Forecast', color='red')
plt.fill_between(future_dates_array, 
                 forecast_lower.flatten(), 
                 forecast_upper.flatten(), 
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Adani Ports Stock Price 6-Month Forecast with Confidence Interval (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
# -------------------------------
# Smooth Forecast for Visualization
# -------------------------------

window_size = 5  # adjust for smoothness; larger = smoother

# Smooth the forecast and confidence intervals
forecast_smooth = pd.Series(forecast.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_upper_smooth = pd.Series(forecast_upper.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_lower_smooth = pd.Series(forecast_lower.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()

# Convert future_dates to numpy array for plotting
future_dates_array = future_dates.to_numpy()

# Plot
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price', color='blue')
plt.plot(future_dates_array, forecast_smooth, label='6-Month Forecast (Smoothed)', color='red')
plt.fill_between(future_dates_array, 
                 forecast_lower_smooth, 
                 forecast_upper_smooth, 
                 color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Adani Ports Stock Price 6-Month Forecast with Smoothed Curve (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# Historical Data
# -------------------------------
dates = df['Date']
prices = df['Close']

# -------------------------------
# ARIMA Forecast (assume forecast_arima exists)
# -------------------------------
# Example: forecast_arima is a numpy array of future prices
# forecast_arima_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=len(forecast_arima))
# Replace with your actual ARIMA forecast

# -------------------------------
# SARIMA Forecast (assume forecast_sarima exists)
# -------------------------------
# Example: forecast_sarima is a numpy array of future prices
# forecast_sarima_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=len(forecast_sarima))
# Replace with your actual SARIMA forecast

# -------------------------------
# Prophet Forecast (assume forecast_prophet_df exists)
# -------------------------------
# forecast_prophet_df has 'ds' and 'yhat' columns
# prophet_dates = forecast_prophet_df['ds']
# prophet_forecast = forecast_prophet_df['yhat']

# -------------------------------
# LSTM Forecast (6-month)
# -------------------------------
future_days = 132
future_dates = pd.bdate_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=future_days)

# Smoothed LSTM forecast and confidence intervals
window_size = 5
forecast_smooth = pd.Series(forecast.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_upper_smooth = pd.Series(forecast_upper.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_lower_smooth = pd.Series(forecast_lower.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()

# -------------------------------
# Plot Combined Forecasts
# -------------------------------
plt.figure(figsize=(18,8))
plt.plot(dates, prices, label='Historical Close Price', color='black')

# Uncomment and replace with your actual forecasts
# plt.plot(forecast_arima_dates, forecast_arima, label='ARIMA Forecast', color='blue')
# plt.plot(forecast_sarima_dates, forecast_sarima, label='SARIMA Forecast', color='green')
# plt.plot(prophet_dates, prophet_forecast, label='Prophet Forecast', color='orange')

plt.plot(future_dates.to_numpy(), forecast_smooth, label='LSTM Forecast (Smoothed)', color='red')
plt.fill_between(future_dates.to_numpy(), forecast_lower_smooth, forecast_upper_smooth, color='red', alpha=0.2, label='LSTM 95% CI')

plt.title('Adani Ports Stock Price Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# %%
pip install mplfinance


# %%
import mplfinance as mpf
import pandas as pd
import numpy as np

# -------------------------------
# Prepare Historical Data for Candles
# -------------------------------
df_candle = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
df_candle.set_index('Date', inplace=True)

# -------------------------------
# Prepare Forecast Data for Candles (LSTM)
# -------------------------------
forecast_candle = pd.DataFrame({
    'Open': forecast_smooth.shift(1).fillna(forecast_smooth.iloc[0]),  # previous close as open
    'High': forecast_upper_smooth,
    'Low': forecast_lower_smooth,
    'Close': forecast_smooth
}, index=future_dates)

# -------------------------------
# Combine Historical + Forecast
# -------------------------------
combined = pd.concat([df_candle, forecast_candle])

# -------------------------------
# Limit to last 1 year historical + 6-month forecast
# -------------------------------
subset = combined[-(252 + len(future_dates)):]  # 252 trading days ≈ 1 year

# -------------------------------
# Plot Candlestick Chart
# -------------------------------
mpf.plot(
    subset,
    type='candle',       # red/green candles
    style='charles',
    figsize=(16,8),
    title='Adani Ports Stock Price with LSTM 6-Month Forecast',
    ylabel='Price',
    volume=False
)


# %%
import mplfinance as mpf
import pandas as pd
import numpy as np

# -------------------------------
# Prepare Historical Data for Candles
# -------------------------------
df_candle = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
df_candle.set_index('Date', inplace=True)

# -------------------------------
# Future Forecast for 90 days
# -------------------------------
future_days = 90
future_dates = pd.bdate_range(df_candle.index[-1] + pd.Timedelta(days=1), periods=future_days)

# Smoothed LSTM forecast and confidence intervals
window_size = 5
forecast_smooth_90 = pd.Series(forecast[:future_days].flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_upper_smooth_90 = pd.Series(forecast_upper[:future_days].flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_lower_smooth_90 = pd.Series(forecast_lower[:future_days].flatten()).rolling(window=window_size, min_periods=1, center=True).mean()

# -------------------------------
# Prepare Forecast Candles
# -------------------------------
forecast_candle = pd.DataFrame({
    'Open': forecast_smooth_90.shift(1).fillna(forecast_smooth_90.iloc[0]),
    'High': forecast_upper_smooth_90,
    'Low': forecast_lower_smooth_90,
    'Close': forecast_smooth_90
}, index=future_dates)

# -------------------------------
# Combine Historical + Forecast
# -------------------------------
combined = pd.concat([df_candle, forecast_candle])

# -------------------------------
# Limit to last 1 year historical + 90-day forecast
# -------------------------------
subset = combined[-(252 + future_days):]  # 252 trading days ≈ 1 year

# -------------------------------
# Plot Candlestick Chart
# -------------------------------
mpf.plot(
    subset,
    type='candle',       # red/green candles
    style='charles',
    figsize=(16,8),
    title='Adani Ports Stock Price with LSTM 90-Day Forecast',
    ylabel='Price',
    volume=False
)


