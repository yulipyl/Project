# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:39:19 2024

@author: leand
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def prepare_data_for_modeling(data, features):
    output_var = data[['direction']]

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(feature_transform, columns=features, index=data.index)

    output_var = output_var.values.ravel()

    return feature_transform, output_var, scaler

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)