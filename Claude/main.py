# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:41:02 2024

@author: leand
"""

from config import *
from data_loader import get_dow30_tickers, load_stock_data
from feature_engineering import feature_engineering
from model_preparation import prepare_data_for_modeling, create_sequences
from model_training import build_and_train_model, evaluate_model
from portfolio_management import backtest_strategy, calculate_performance
import yfinance as yf

def main():
    dow30_tickers = get_dow30_tickers()
    models = {}
    scalers = {}

    for ticker in dow30_tickers:
        print(f"Processing {ticker}...")
        try:
            data = load_stock_data(ticker, START_DATE, END_DATE)
            data = feature_engineering(data)
           
            features, target, scaler = prepare_data_for_modeling(data, FEATURES)
            X, y = create_sequences(features, target, time_steps=10)

            model = build_and_train_model(X, y)
            accuracy, precision, recall, f1 = evaluate_model(model, X, y)

            print(f'{ticker} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            models[ticker] = model
            scalers[ticker] = scaler

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("Finished processing all tickers.")

    portfolio_history = backtest_strategy(
        models,
        scalers,
        dow30_tickers,
        BACKTEST_START_DATE,
        BACKTEST_END_DATE,
        INITIAL_INVESTMENT
    )

    print("Portfolio History:")
    for entry in portfolio_history:
        print(f"Date: {entry['date']}, Value: ${entry['value']:.2f}")

    # Download benchmark data (assuming S&P 500 as benchmark)
    benchmark_data = yf.download('^GSPC', start=BACKTEST_START_DATE, end=BACKTEST_END_DATE)

    performance = calculate_performance(portfolio_history, benchmark_data)
    
    print("\nPerformance Metrics:")
    print(f"Portfolio Return: {performance['Portfolio Return']:.2%}")
    print(f"Benchmark Return: {performance['Benchmark Return']:.2%}")
    print(f"Alpha: {performance['Alpha']:.2%}")

if __name__ == "__main__":
    main()