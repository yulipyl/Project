# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:38:03 2024

@author: leand
"""

import numpy as np
import ta

def feature_engineering(data):
    print(f"Initial data shape: {data.shape}")

    data['macd'] = ta.trend.macd(data['Adj Close']).fillna(0)

    bollinger = ta.volatility.BollingerBands(data['Adj Close'])
    data['boll_ub'] = bollinger.bollinger_hband().fillna(data['Adj Close'])
    data['boll_lb'] = bollinger.bollinger_lband().fillna(data['Adj Close'])

    data['rsi_30'] = ta.momentum.RSIIndicator(data['Adj Close'], window=30).rsi().fillna(50)

    if len(data) >= 30:
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Adj Close'], window=30)
        data['dx_30'] = adx.adx().fillna(20)
    else:
        data['dx_30'] = np.nan

    data['close_30_sma'] = data['Adj Close'].rolling(window=30).mean().ffill()
    data['close_60_sma'] = data['Adj Close'].rolling(window=60).mean().bfill()

    data.dropna(inplace=True)
    print(f"After dropping NaN values: {data.shape}")
    
    return data