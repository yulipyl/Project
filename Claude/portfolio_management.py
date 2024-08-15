# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:40:20 2024

@author: leand
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data_loader import load_stock_data
from feature_engineering import feature_engineering

def predict_stock_direction(model, scaler, data, features):
    scaled_features = scaler.transform(data[features])
    scaled_features = scaled_features.reshape((1, scaled_features.shape[0], scaled_features.shape[1]))
    prediction = model.predict(scaled_features)
    return 1 if prediction > 0.5 else 0

def generate_signals(models, scalers, tickers, date):
    signals = {}
    for ticker in tickers:
        try:
            data = load_stock_data(ticker, date, (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
            data = feature_engineering(data)
            if not data.empty:
                signals[ticker] = predict_stock_direction(models[ticker], scalers[ticker], data, FEATURES)
        except Exception as e:
            print(f"Error generating signal for {ticker}: {e}")
    return signals

def allocate_portfolio(signals, total_investment):
    bullish_stocks = [ticker for ticker, signal in signals.items() if signal == 1]
    if not bullish_stocks:
        return {ticker: 0 for ticker in signals}
    allocation_per_stock = total_investment / len(bullish_stocks)
    return {ticker: allocation_per_stock if signal == 1 else 0 for ticker, signal in signals.items()}

def calculate_returns(portfolio, price_data):
    total_return = 0
    for ticker, investment in portfolio.items():
        if investment > 0:
            returns = price_data[ticker]['returns'].iloc[-1]
            total_return += investment * returns
    return total_return

def backtest_strategy(models, scalers, tickers, start_date, end_date, initial_investment):
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    portfolio = {}
    portfolio_value = initial_investment
    portfolio_history = []
    
    while current_date <= end_date:
        signals = generate_signals(models, scalers, tickers, current_date.strftime("%Y-%m-%d"))
        portfolio = allocate_portfolio(signals, portfolio_value)
        
        next_date = current_date + timedelta(days=7)
        price_data = {ticker: load_stock_data(ticker, current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")) 
                      for ticker in tickers}
        
        returns = calculate_returns(portfolio, price_data)
        portfolio_value += returns
        
        portfolio_history.append({
            'date': current_date.strftime("%Y-%m-%d"),
            'portfolio': portfolio.copy(),
            'value': portfolio_value
        })
        
        current_date = next_date
    
    return portfolio_history

def calculate_performance(portfolio_history, benchmark_data):
    portfolio_returns = pd.Series([h['value'] for h in portfolio_history]).pct_change()
    benchmark_returns = benchmark_data['Adj Close'].pct_change()
    
    portfolio_cumulative_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    benchmark_cumulative_return = (1 + benchmark_returns).cumprod().iloc[-1] - 1
    
    alpha = portfolio_cumulative_return - benchmark_cumulative_return
    
    return {
        'Portfolio Return': portfolio_cumulative_return,
        'Benchmark Return': benchmark_cumulative_return,
        'Alpha': alpha
    }