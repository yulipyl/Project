# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:32:28 2024

@author: leand
"""

## Step 1 - Import the Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import ta  # Technical Analysis library
from datetime import datetime, timedelta


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

## Step 2 – Define the training time frame
start_date = "2014-01-01"
end_date = "2024-01-01"

## Step 3 - Helper Functions

def get_dow30_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components'
    tables = pd.read_html(url)
    dow30_table = tables[1]
    tickers = dow30_table['Symbol'].tolist()
    return tickers

def load_stock_data(ticker, start_date, end_date):
    ticker = ticker.replace('.', '-')
    data = yf.download(ticker, start=start_date, end=end_date)
    data['returns'] = data['Adj Close'].pct_change().fillna(0)
    data['direction'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
    return data

def feature_engineering(data):
    # Check if there's enough data to apply indicators
    min_length = max(30, 60)  # Maximum of any window size used in indicators
    if len(data) < min_length:
        raise ValueError(f"Not enough data to apply feature engineering. Need at least {min_length} rows.")

    data['macd'] = ta.trend.macd(data['Adj Close'])

    bollinger = ta.volatility.BollingerBands(data['Adj Close'])
    data['boll_ub'] = bollinger.bollinger_hband()
    data['boll_lb'] = bollinger.bollinger_lband()

    data['rsi_30'] = ta.momentum.RSIIndicator(data['Adj Close'], window=30).rsi()

    # Check if data length is sufficient for ADX calculation
    if len(data) >= 30:
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Adj Close'], window=30)
        data['dx_30'] = adx.adx()
    else:
        data['dx_30'] = np.nan

    data['close_30_sma'] = data['Adj Close'].rolling(window=30).mean()
    data['close_60_sma'] = data['Adj Close'].rolling(window=60).mean()
    data.dropna(inplace=True)
    
    return data


def prepare_data_for_modeling(data, features):
    output_var = data[['direction']]

    # Initialize and fit the scaler
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(feature_transform, columns=features, index=data.index)

    # The target variable does not need to be scaled the same way as features
    output_var = output_var.values.ravel()  # Flatten to 1D array

    train_size = int(len(feature_transform) * 0.8)
    X_train, X_test = feature_transform[:train_size], feature_transform[train_size:]
    X_train = X_train.to_numpy()  # Convert DataFrame to numpy array
    X_test = X_test.to_numpy()    # Convert DataFrame to numpy array
    y_train, y_test = output_var[:train_size], output_var[train_size:]

    X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler


## Step 4 - Loop through Each Ticker

dow30_tickers = get_dow30_tickers()
models = {}
scalers = {}

features = ['Adj Close', 'Volume', 'boll_ub', 'boll_lb', 'rsi_30', 'close_30_sma', 'close_60_sma']

for ticker in dow30_tickers:
    print(f"Processing {ticker}...")
    try:
        data = load_stock_data(ticker, start_date, end_date)
        data = feature_engineering(data)
        
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_modeling(data, features)

        ## Build and Train the LSTM Model
        lstm = Sequential()
        lstm.add(LSTM(32, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=False))
        lstm.add(Dense(1, activation='sigmoid'))
        lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False, validation_data=(X_test, y_test), callbacks=[early_stopping])

        ## Make Predictions
        y_pred = (lstm.predict(X_test) > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f'{ticker} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        models[ticker] = lstm
        scalers[ticker] = scaler

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print("Finished processing all tickers.")

testing = load_stock_data("AAPL", start_date, end_date)
testing = feature_engineering(data)
testing
## Step 5 - Generate Future Predictions

def generate_predictions(models, scalers, tickers, future_data):
    predictions = {}
    for ticker in tickers:
        model = models.get(ticker)
        scaler = scalers.get(ticker)
        if model and scaler:
            try:
                data = future_data[ticker]
                if len(data) < 60:
                    print(f"Insufficient data for {ticker}. Need at least 60 rows after feature engineering.")
                    continue

                # Apply feature engineering
                data = feature_engineering(data)
                if len(data) < 60:
                    print(f"Insufficient data for {ticker} after feature engineering. Skipping.")
                    continue

                future_features = data[features]
                
                # Convert DataFrame to array if necessary
                if isinstance(future_features, pd.DataFrame):
                    future_features = future_features.to_numpy()
                
                # Transform features using the scaler
                if future_features.shape[1] != len(features):
                    raise ValueError(f"Feature mismatch for {ticker}: Expected {len(features)} features, but got {future_features.shape[1]}.")

                future_features_scaled = scaler.transform(future_features)
                future_features_scaled = np.array(future_features_scaled).reshape((future_features_scaled.shape[0], 1, future_features_scaled.shape[1]))

                # Predict probabilities
                predictions[ticker] = model.predict(future_features_scaled).flatten()
            except Exception as e:
                print(f"Error processing future data for {ticker}: {e}")
    return predictions


## Step 6 - Generate Trading Signals

def generate_trading_signals(predictions):
    signals = {}
    for ticker, pred_probs in predictions.items():
        # Compute average confidence score
        avg_confidence = np.mean(pred_probs)
        signals[ticker] = avg_confidence
    return signals

## Step 7 - Portfolio Allocation

def allocate_portfolio(signals, total_investment):
    allocations = {}
    # Normalize confidence scores
    total_confidence = sum(signals.values())
    
    if total_confidence > 0:
        for ticker, confidence in signals.items():
            # Allocate proportionally based on confidence
            allocations[ticker] = (confidence / total_confidence) * total_investment
    return allocations

## Step 8 - Simulate Portfolio Performance

def simulate_portfolio_performance(allocations, historical_data):
    total_investment = sum(allocations.values())
    portfolio_value = 0

    for ticker, amount in allocations.items():
        if ticker in historical_data:
            stock_data = historical_data[ticker]
            # Simulate buying the stock at the latest available price
            buy_price = stock_data['Adj Close'][-1]
            # Assume selling at the future predicted price (example: latest available future price)
            sell_price = stock_data['Adj Close'][-1] * 1.02  # Example: 2% profit
            portfolio_value += (sell_price - buy_price) * (amount / buy_price)

    return portfolio_value

# Example future data (replace with actual data)
fut_date = "2024-08-01"

future_data = {ticker: load_stock_data(ticker, end_date, fut_date) for ticker in dow30_tickers}
predictions = generate_predictions(models, scalers, dow30_tickers, future_data)
signals = generate_trading_signals(predictions)

# Portfolio Allocation
investment_amount = 100000  # Example investment amount
allocations = allocate_portfolio(signals, investment_amount)

# Simulate portfolio 
historical_data = {ticker: load_stock_data(ticker, start_date, end_date) for ticker in dow30_tickers}
portfolio_value = simulate_portfolio_performance(allocations, historical_data)

print(f"Allocated Portfolio: {allocations}")
print(f"Simulated Portfolio Value: ${portfolio_value:.2f}")

