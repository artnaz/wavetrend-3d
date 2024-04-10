# _example.py
# shows a practical example how to use the library
# Copyright (C) 2024 Arthur Nazarian

import ccxt   # for illustration purposes, not included in dependencies
import polars as pl

from src.wavetrend_3d.plotting import PlotWaveTrend3D
from src.wavetrend_3d.wavetrend_3d import WaveTrend3D

# --- 1. Get data --- #
binance = ccxt.binance()
df = pl.LazyFrame(
    binance.fetch_ohlcv('BTC/USDT', '1h'),
    schema=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
df = (
    df
    .with_columns(
        pl.col(pl.FLOAT_DTYPES).shrink_dtype().round(0),  # to minimise storage requirements
        pl.from_epoch('timestamp', time_unit='ms')  # datetime datatype is easier to work with
    )
    .rename({'timestamp': 'datetime'})
).collect()
print(df)
df.write_parquet('sample_btc_data.parquet')


# --- 2. Calculate WaveTrend3D oscillator series --- #
wt3d = WaveTrend3D(df['close'])
wt3d.compute_series()
wt3d.compute_signals()
print('Fast oscillator\'s shape:', wt3d.series_fast.shape)


# --- 3. Plot WaveTrend3D --- #
plot_mirror = True  # to easily switch whether to add a mirrored version

if plot_mirror:
    # set the proportions for the price, regular oscillator, and mirrored oscillator
    plot = PlotWaveTrend3D([.7, .2, .1], height=800)
else:
    # set the proportions for the price and regular oscillator
    plot = PlotWaveTrend3D([.8, .2], height=800)

plot.add_candles(df, 'BTC-USDT', '1h')
plot.add_wavetrend(wt3d, main=True, mirror=plot_mirror)
plot.show()
