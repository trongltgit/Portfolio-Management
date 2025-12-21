# user/chart/services.py
import pandas as pd
import yfinance as yf

def get_historical_data(symbol, period="6mo", interval="1d"):
    """
    Lấy dữ liệu lịch sử từ Yahoo Finance
    period: 1mo, 3mo, 6mo, 1y
    interval: 1d, 1wk, 1mo
    """
    data = yf.download(symbol, period=period, interval=interval)
    data.reset_index(inplace=True)
    return data

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Thêm cột MACD, Signal
    """
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    return df
