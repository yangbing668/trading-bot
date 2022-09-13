import os
import math
import logging
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tkinter import _flatten

def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

# def get_state(data, t, n_days):
#     """Returns an n-day state representation ending at time t
#     """
#     d = t - n_days + 1
#     block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
#     res = []
#     for i in range(n_days - 1):
#         res.append(sigmoid(block[i + 1] - block[i]))
#     return np.array([res])

def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    #blocks = []
    #for feature in datan.columns:
    #feature = "Close"
    #data = list(datan[feature])
    if len(data) == 0:
        raise Exception("DATA IS ZERO!!! CHECK YFINANCE OUTPUT")
    d = t - n_days - 1
    if d >= 0:
        block = data[d: t + 1]
    else:
        block = data[0:d] # pad with
    res = []
    for i in range(n_days - 1):
        x = sigmoid(block[i + 1] - block[i]) # x is number
        res.append(x)
    return np.array([res])

