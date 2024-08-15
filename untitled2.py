# -- coding: utf-8 --

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
    print(f"Initial data shape: {data.shape}")

    # Applying MACD
    data['macd'] = ta.trend.macd(data['Adj Close']).fillna(0)

    # Applying Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Adj Close'])
    data['boll_ub'] = bollinger.bollinger_hband().fillna(data['Adj Close'])
    data['boll_lb'] = bollinger.bollinger_lband().fillna(data['Adj Close'])

    # Applying RSI
    data['rsi_30'] = ta.momentum.RSIIndicator(data['Adj Close'], window=30).rsi().fillna(50)

    # Applying ADX
    if len(data) >= 30:
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Adj Close'], window=30)
        data['dx_30'] = adx.adx().fillna(20)
    else:
        data['dx_30'] = np.nan

    # SMA
    data['close_30_sma'] = data['Adj Close'].rolling(window=30).mean().fillna(method='backfill')
    data['close_60_sma'] = data['Adj Close'].rolling(window=60).mean().fillna(method='backfill')

    # Dropping NaN values
    data.dropna(inplace=True)
    print(f"After dropping NaN values: {data.shape}")
    
    return data

def prepare_data_for_modeling(data, features):
    output_var = data[['direction']]

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(feature_transform, columns=features, index=data.index)

    output_var = output_var.values.ravel()

    train_size = int(len(feature_transform) * 0.8)
    X_train, X_test = feature_transform[:train_size], feature_transform[train_size:]
    y_train, y_test = output_var[:train_size], output_var[train_size:]

    X_train = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))

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
models.get("AAPL")

## Step 5 - Generate Future Predictions
def generate_predictions(models, scalers, tickers, future_end_date):
    predictions = {}
    for ticker in tickers:
        model = models.get(ticker)
        scaler = scalers.get(ticker)
        if model and scaler:
            try:
                # Load historical data up to the end of the current week
                historical_data_ticker = load_stock_data(ticker, start_date, end_date)
                new_data = load_stock_data(ticker, "2024-01-01", future_end_date)
                
                # Append the new data to the historical data
                combined_data = pd.concat([historical_data_ticker, new_data])
                
                # Apply feature engineering to the combined dataset
                combined_data = feature_engineering(combined_data)
                
                # Ensure we have enough data after feature engineering
                if len(combined_data) < 60:
                    print(f"Insufficient data for {ticker} after feature engineering. Skipping.")
                    continue
                
                future_features = combined_data[features]
                
                # Transform features using the scaler
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
        avg_confidence = np.mean(pred_probs)
        signals[ticker] = avg_confidence
    return signals

## Step 7 - Portfolio Allocation
def allocate_portfolio(signals, total_investment):
    allocations = {}
    total_confidence = sum(signals.values())
   
    if total_confidence > 0:
        for ticker, confidence in signals.items():
            allocations[ticker] = (confidence / total_confidence) * total_investment
    return allocations

# Ensure date formatting is correct
def initialize_portfolio(initial_date, models, scalers, tickers, start_date, end_date, initial_investment):
    historical_data = {}
    for ticker in tickers:
        try:
            data = load_stock_data(ticker, start_date, end_date)
            historical_data[ticker] = feature_engineering(data)
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
    
    predictions = generate_predictions(models, scalers, tickers, initial_date)
    signals = generate_trading_signals(predictions)
    allocations = allocate_portfolio(signals, initial_investment)
    return allocations, predictions

# Rebalance Portfolio Weekly
def rebalance_portfolio(start_date, end_date, initial_portfolio, initial_predictions, models, scalers, tickers, historical_data):
    current_portfolio = initial_portfolio.copy()
    portfolio_history = []
    weekly_returns = []
    
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_date:
        week_end_date = current_date + timedelta(days=7)
        
        predictions = generate_predictions(models, scalers, tickers, week_end_date.strftime("%Y-%m-%d"))
        signals = generate_trading_signals(predictions)
        
        # Calculate portfolio returns based on new predictions
        current_returns = 0
        for ticker in current_portfolio.keys():
            if ticker in signals:
                new_signal = signals[ticker]
                old_signal = initial_predictions[ticker]
                
                # Sell if direction changes or buy more if confidence increases
                if new_signal < 0.5 and old_signal >= 0.5:
                    current_returns += current_portfolio[ticker]  # Sell
                    current_portfolio[ticker] = 0
                elif new_signal > old_signal:
                    additional_investment = current_returns * 0.1  # Reinvest 10% of returns
                    current_portfolio[ticker] += additional_investment
                    current_returns -= additional_investment
        
        weekly_returns.append(current_returns)
        portfolio_history.append(current_portfolio.copy())
        
        # Move to the next week
        current_date = week_end_date
    
    return portfolio_history, weekly_returns



initial_investment = 1000000  # Example investment of $1,000,000
simulation_start_date = "2024-01-07"
simulation_end_date = "2024-08-01"

# Initialize historical data
historical_data = {}
for ticker in dow30_tickers:
    try:
        data = load_stock_data(ticker, start_date, end_date)
        historical_data[ticker] = feature_engineering(data)
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")

# Initialize portfolio and predictions
initial_portfolio, initial_predictions = initialize_portfolio(
    simulation_start_date,
    models,
    scalers,
    dow30_tickers,
    start_date="2014-01-01",
    end_date="2024-01-01",
    initial_investment=initial_investment
)

# Run Rebalancing
portfolio_history, weekly_returns = rebalance_portfolio(
    simulation_start_date,
    simulation_end_date,
    initial_portfolio,
    initial_predictions,
    models,
    scalers,
    dow30_tickers,  # Ensure this is the correct list of tickers
    historical_data  # Pass the historical data here
)

# Print Results
print("Portfolio History:")
for i, week in enumerate(portfolio_history):
    print(f"Week {i+1}: {week}")

print("Weekly Returns:")
for i, ret in enumerate(weekly_returns):
    print(f"Week {i+1}: ${ret:.2f}")

# Print Results
print("Portfolio History:")
for i, week in enumerate(portfolio_history):
    print(f"Week {i+1}: {week}")

print("Weekly Returns:")
for i, ret in enumerate(weekly_returns):
    print(f"Week {i+1}: ${ret:.2f}")
