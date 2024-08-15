# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:00:14 2024

@author: leand
"""

## Step 1 - Import the Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

## Step 2 – Reading our training data and getting our training data in shape
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    # data.reset_index(inplace=True)
    data.drop(['Adj Close'], axis=1, inplace=True)
    return data

def feature_engineering(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean() 
    data.dropna(inplace=True)
    return data

# Define the stock ticker and the training time frame
stock_ticker = "AAPL"
start_date = "2010-01-01"  # Start with a long enough historical period
end_date = "2024-01-01"   

# Load and prepare the training data
dataset_train = load_stock_data(stock_ticker, start_date, end_date)
dataset_train = feature_engineering(dataset_train)

#Print the shape of Dataframe  and Check for Null Values
print("Dataframe Shape: ", dataset_train.shape)
print("Null Value Present: ", dataset_train.isnull().values.any())

## Step 3 - Setting the Target Variable, Selecting the Features, and Scaling
output_var = pd.DataFrame(dataset_train['Close'])
features = ['Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_50']

scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(dataset_train[features])
output_var_scaled = scaler.fit_transform(output_var)  # Scaling the target variable

# Convert back to DataFrame
feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=dataset_train.index)
output_var_scaled = pd.DataFrame(columns=['Close'], data=output_var_scaled, index=dataset_train.index)

##Step 4 - Splitting to Training set and Test set
train_size = int(len(feature_transform) * 0.8)
X_train, X_test = feature_transform[:train_size], feature_transform[train_size:]
y_train, y_test = output_var_scaled[:train_size].values.ravel(), output_var_scaled[train_size:].values.ravel()

X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

## Step 5 - Building and Training the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

## Step 6 - Making the Prediction
y_pred = lstm.predict(X_test)

def predict_future_prices(model, last_known_data, days_to_predict, scaler, feature_columns):
    predictions = []
    current_data = last_known_data

    for _ in range(days_to_predict):
        # Predict the next day's price
        pred = model.predict(current_data)
        predictions.append(pred[0, 0])
        
        # Prepare the new data point for the next prediction
        # Remove the first row and append the predicted value
        current_data = np.roll(current_data, shift=-1, axis=1)
        current_data[0, -1, 0] = pred

    return predictions

days_to_predict = 10
last_known_data = np.array(feature_transform[-1:]).reshape((1, 1, len(features)))
future_predictions = predict_future_prices(lstm, last_known_data, days_to_predict, scaler, features)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_predictions = future_predictions.flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE on Test Set: {mse:.4f}')
print(f'R2 Score on Test Set: {r2:.4f}')

# Prepare dates for plotting
last_date = dataset_train.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)

# Prepare historical dates for plotting
historical_dates = dataset_train.index
test_dates = historical_dates[-len(y_test):]
predicted_dates = historical_dates[-len(y_pred):]

# Plot historical and predicted data
plt.figure(figsize=(14, 7))
plt.plot(historical_dates, dataset_train['Close'], label='Historical Data', color='blue')
plt.plot(test_dates, scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Values', color='green')
plt.plot(predicted_dates, scaler.inverse_transform(y_pred.reshape(-1, 1)), label='LSTM Predictions', color='red')
plt.plot(future_dates, future_predictions, linestyle='--', color='orange', label='Future Predictions')

# Add labels and legend
plt.title(f'Stock Price Prediction for {stock_ticker} with Extended Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Zoomed-In Plot for the Last 15 Days
plt.figure(figsize=(14, 7))

# Extracting the last 15 days data
zoom_start_date = dataset_train.index[-15]
zoom_end_date = dataset_train.index[-1]
zoom_dates = dataset_train.index[-15:]
zoom_true_values = scaler.inverse_transform(y_test[-15:].reshape(-1, 1))
zoom_lstm_predictions = scaler.inverse_transform(y_pred[-15:].reshape(-1, 1))
zoom_future_predictions = future_predictions[:15]  # Only the first 15 future days

plt.plot(zoom_dates, scaler.inverse_transform(output_var_scaled.loc[zoom_dates].values), label='True Values', color='green')
plt.plot(zoom_dates, zoom_lstm_predictions, label='LSTM Predictions', color='red')
plt.plot(future_dates[:15], zoom_future_predictions, linestyle='--', color='orange', label='Future Predictions')

# Add labels and legend
plt.title(f'Zoomed-In View: Last 15 Days with LSTM Predictions for {stock_ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()