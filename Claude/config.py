# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:37:07 2024

@author: leand
"""

# Configuration parameters
START_DATE = "2014-01-01"
END_DATE = "2024-01-01"
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2024-08-15"  # Current date
INITIAL_INVESTMENT = 1000000  # $1,000,000

FEATURES = ['Adj Close', 'Volume', 'boll_ub', 'boll_lb', 'rsi_30', 'close_30_sma', 'close_60_sma']