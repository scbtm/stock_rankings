import polars as pl
import numpy as np

from typing import List

#Technical Indicators and other helper functions to be used in the pipeline

#Simple Moving Average
def moving_averages(df: pl.DataFrame) -> pl.DataFrame: 
    df = df.with_columns([
            (pl.col('Close').rolling_mean(5)).alias('5d_SMA'),
            (pl.col('Close').rolling_mean(10)).alias('10d_SMA'),
            (pl.col('Close').rolling_mean(20)).alias('20d_SMA'),
            (pl.col('Close').rolling_mean(50)).alias('50d_SMA'),
            (pl.col('Close').rolling_mean(100)).alias('100d_SMA'),
            (pl.col('Close').rolling_mean(200)).alias('200d_SMA')
            ])
                    
    reversed_timepoints = [5, 10, 20, 50, 100, 200][::-1]
    weights = np.array(reversed_timepoints)/sum(reversed_timepoints)
                    
    df = df.with_columns([
            ((pl.col('Close')/pl.col('5d_SMA'))*weights[0]).alias('5d_ratio'),
            ((pl.col('Close')/pl.col('10d_SMA'))*weights[1]).alias('10d_ratio'),
            ((pl.col('Close')/pl.col('20d_SMA'))*weights[2]).alias('20d_ratio'),
            ((pl.col('Close')/pl.col('50d_SMA'))*weights[3]).alias('50d_ratio'),
            ((pl.col('Close')/pl.col('100d_SMA'))*weights[4]).alias('100d_ratio'),
            ((pl.col('Close')/pl.col('200d_SMA'))*weights[5]).alias('200d_ratio')
            ])

    df = df.drop(['5d_SMA', '10d_SMA', '20d_SMA', '50d_SMA', '100d_SMA', '200d_SMA'])

    #Weighted Average of Ratios
    w_avg_ratios = [pl.col('5d_ratio'),
                    pl.col('10d_ratio'),
                    pl.col('20d_ratio'),
                    pl.col('50d_ratio'),
                    pl.col('100d_ratio'),
                    pl.col('200d_ratio')]

    df = df.with_columns([
        pl.sum_horizontal(w_avg_ratios).alias('Close_weighted_moving_avg_ratios')
    ])

    df = df.drop(['5d_ratio', '10d_ratio', '20d_ratio', '50d_ratio', '100d_ratio', '200d_ratio'])

    return df

#Relative Strength Index
def rsi(df: pl.DataFrame) -> pl.DataFrame:
    delta = df['Close'].pct_change()
    gain = delta.clip(lower_bound=0).rolling_mean(15)
    loss = (-delta).clip(lower_bound=0).rolling_mean(15)
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df = df.with_columns(rsi.alias('RSI'))

    return df

#Bollinger Bands
def bollinger_indicator(df: pl.DataFrame) -> pl.DataFrame:
    sma = df['Close'].rolling_mean(15)
    std_dev = df['Close'].rolling_std(15)
    upper_band = (sma + 2 * std_dev).alias('upper_band')
    lower_band = (sma - 2 * std_dev).alias('lower_band')

    bollinger_interval_ratio = ((upper_band - lower_band)/sma).alias('bollinger_interval_ratio')

    df = df.with_columns(bollinger_interval_ratio)

    return df

#Stochastic oscillator
def stochastic_oscillator(df: pl.DataFrame) -> pl.DataFrame:

    lowest_low = df['Low'].rolling_min(15)
    highest_high = df['High'].rolling_max(15)
    k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    df = df.with_columns(k_percent.alias('stochastic_oscillator'))

    return df

#ADX
def adx(df: pl.DataFrame, period:int = 15) -> pl.DataFrame:
    df = df.with_columns([
         (df['High'] - df['Low']).ewm_mean(span = period).alias('true_range'),
         (df['High'] - df['High'].shift(1)).abs().alias('high_diff'),
         (df['Low'] - df['Low'].shift(1)).abs().alias('low_diff')
         ])
                
    df = df.with_columns([
         (pl.when((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0))).then(df['high_diff']).otherwise(0).alias('plus_dm'),
         (pl.when((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0))).then(df['low_diff']).otherwise(0).alias('minus_dm')
         ])
                
    df = df.with_columns([
        (df['true_range'].ewm_mean(span = period)).alias('atr'),
        (df['plus_dm'].ewm_mean(span = period)).alias('plus_dm_ewma'),
        (df['minus_dm'].ewm_mean(span = period)).alias('minus_dm_ewma')
    ])
                
    df = df.with_columns([
        ((df['plus_dm_ewma'] / df['atr']) * 100).alias('plus_di'),
        ((df['minus_dm_ewma'] / df['atr']) * 100).alias('minus_di')
    ])

    df = df.with_columns([
        (( (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di']) ) * 100).alias('adx')
    ])
                
    #this are columns used in intermediate calculations used to build the ADX. We only keep atr as this is another useful indicator
    cols_to_drop = ['true_range', 'high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'plus_dm_ewma', 'minus_dm_ewma', 'plus_di', 'minus_di']
    df = df.drop(cols_to_drop)

    return df

#CCI
def cci(df: pl.DataFrame, period:int = 15) -> pl.DataFrame:

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    mean_deviation = (( typical_price - typical_price.rolling_mean(period) ).abs()).ewm_mean(span = period)
    cci = (typical_price - typical_price.rolling_mean(period)) / (0.015 * mean_deviation)
    df = df.with_columns(cci.alias('CCI'))

    return df

#Ichiomoku Cloud
def ichimoku_cloud(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26, period=15):
    tenkan_sen = (df['High'].rolling_max(tenkan_period) + df['Low'].rolling_min(tenkan_period)) / 2

    kijun_sen = (df['High'].rolling_max(kijun_period) + df['Low'].rolling_min(kijun_period)) / 2

    tenkan_kijun = (tenkan_sen - kijun_sen)/kijun_sen

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    senkou_span_b = ((df['High'].rolling_max(senkou_span_b_period) + df['Low'].rolling_min(senkou_span_b_period)) / 2).shift(displacement)

    senkou_cloud = (senkou_span_a - senkou_span_b)/senkou_span_b

    chikou_span = df['Close'].shift(displacement)

    chikou_tenkan = (df['Close'] - tenkan_sen)/tenkan_sen
    chikou_kijun = (df['Close'] - kijun_sen)/kijun_sen
    chikou_senkou_a = (df['Close'] - senkou_span_a)/senkou_span_a
    chikou_senkou_b = (df['Close'] - senkou_span_b)/senkou_span_b

    chinkou = (chikou_tenkan + chikou_kijun + chikou_senkou_a + chikou_senkou_b)/4



    df = df.with_columns([
        tenkan_kijun.alias('tenkan_kijun'),
        senkou_cloud.alias('senkou_cloud'),
        chinkou.alias('chinkou')
    ])
    return df