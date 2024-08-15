# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:37:57 2024

@author: leand
"""

import yfinance as yf
import pandas as pd
import numpy as np

def get_dow30_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average#Components'
    tables = pd.read_html(url)
    dow30_table = tables[1]
    return dow30_table['Symbol'].tolist()

def load_stock_data(ticker, start_date, end_date):
    ticker = ticker.replace('.', '-')
    data = yf.download(ticker, start=start_date, end=end_date)
    data['returns'] = data['Adj Close'].pct_change().fillna(0)
    data['direction'] = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, 0)
    return data