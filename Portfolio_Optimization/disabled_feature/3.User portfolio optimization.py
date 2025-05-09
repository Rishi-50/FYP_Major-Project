'''
This feature is currently disabled/commented out.

Original code starts below:
'''

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow can be imported, otherwise use fallback prediction
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
    
    # Only configure TensorFlow if it loaded successfully
    try:
        # Configure TensorFlow for better performance
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            # Limit CPU usage if no GPU
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
    except:
        # If configuration fails, continue without it
        pass
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Fallback prediction function that doesn't require TensorFlow
def simple_prediction(data, ticker, future_days=30):
    """Simple prediction model that's used when TensorFlow has issues"""
    prices = data[ticker].values
    last_price = prices[-1]
    
    # Calculate recent trend (last 30 days)
    if len(prices) >= 30:
        recent_trend = (prices[-1] / prices[-30]) - 1
    else:
        recent_trend = 0.001  # Default small positive trend
    
    # Generate future prices based on simple trend
    predicted_prices = []
    for i in range(future_days):
        # Dampen the trend over time
        dampen_factor = np.exp(-i * 0.05)
        next_price = last_price * (1 + (recent_trend * dampen_factor/30))
        predicted_prices.append(next_price)
        last_price = next_price
    
    # Create future dates
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
    
    return future_dates, np.array(predicted_prices).reshape(-1, 1)

# Rest of the file would be copied here... 