import numpy as np
import pandas as pd

def calculate_ma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    ma = calculate_ma(data, window)
    std = data.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, ma, lower_band

def calculate_stochastic(data, k_period=14, d_period=3):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_adx(data, period=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def calculate_obv(data):
    obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    return obv

def calculate_ichimoku(data):
    high = data['High']
    low = data['Low']
    
    tenkan_window = 9
    kijun_window = 26
    senkou_span_b_window = 52
    
    tenkan_sen = (high.rolling(window=tenkan_window).max() + 
                 low.rolling(window=tenkan_window).min()) / 2
    
    kijun_sen = (high.rolling(window=kijun_window).max() + 
                 low.rolling(window=kijun_window).min()) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)
    
    senkou_span_b = ((high.rolling(window=senkou_span_b_window).max() + 
                      low.rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_window)
    
    chikou_span = data['Close'].shift(-kijun_window)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_fibonacci_retracement(data):
    max_price = data['High'].max()
    min_price = data['Low'].min()
    diff = max_price - min_price
    
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    retracements = []
    
    for level in levels:
        retracements.append(max_price - diff * level)
    
    return retracements

def prepare_technical_indicators(data):
    """Calculate all technical indicators for the given data"""
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = calculate_ma(df['Close'], 5)
    df['MA10'] = calculate_ma(df['Close'], 10)
    df['MA20'] = calculate_ma(df['Close'], 20)
    df['MA50'] = calculate_ma(df['Close'], 50)
    df['MA200'] = calculate_ma(df['Close'], 200)
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # Stochastic Oscillator
    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(df)
    
    # ADX
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df)
    
    # OBV
    df['OBV'] = calculate_obv(df)
    
    # Ichimoku Cloud
    (df['Tenkan_Sen'], df['Kijun_Sen'], df['Senkou_Span_A'],
     df['Senkou_Span_B'], df['Chikou_Span']) = calculate_ichimoku(df)
    
    # Fibonacci Retracement
    df['Fib_Levels'] = [calculate_fibonacci_retracement(df)] * len(df)
    
    return df

def add_technical_indicators(hist_data):
    """Add all technical indicators to the historical data"""
    return prepare_technical_indicators(hist_data)