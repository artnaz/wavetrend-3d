# build_alpha.py
# Copyright (C) 2024 Arthur Nazarian


# You need to import the class WaveTrend3D. This needs to be correctly applied in the platform.
from src.wavetrend_3d.wavetrend_3d_single_file import WaveTrend3D
import polars as pl  # WaveTrend3D uses this instead of Pandas

# from pathlib import Path
# from src.wavetrend_3d.plotting import PlotWaveTrend3D


def ReadData():
    """
    ReadData() follows the Build Alpha example: http://www.buildalpha.com/using-python-in-buildalpha/
    """
    global df1

    # Get data [for Arthur's testing]
    # data_path = Path.cwd().resolve().parents[1]
    # df1 = pl.read_parquet(data_path / 'tests/data/sample_btc_data.parquet')

    # Get data in Build Alpha: they seem to be using pandas (`import pandas as pd`)
    # WaveTrend3D uses Polars, so you need to convert it this way:
    df1 = pl.from_pandas(df_pandas)  # Assuming `df_pandas` is loaded the usual way

    # Or better, if there are no nuances and it works, read the CSV directly
    # df1 = pl.read_csv(file_name_base, ...)




def GetCustomSignal():
    """
    GetCustomSignal() follows the Build Alpha example: http://www.buildalpha.com/using-python-in-buildalpha/
    Of course we apply it to our Polars DataFrame.

    And you can adjust the signal how you want. These are the available signals:
    * wt3d.fast_norm_cross_bullish
    * wt3d.fast_norm_cross_bearish
    * wt3d.divergence_bullish
    * wt3d.divergence_bearish
    * wt3d.fast_median_cross_bullish
    * wt3d.fast_median_cross_bearish
    * wt3d.norm_median_cross_bullish
    * wt3d.norm_median_cross_bearish
    * wt3d.slow_median_cross_bullish
    * wt3d.slow_median_cross_bearish
    """
    ReadData()

    # Instantiate and run the indicator
    wt3d = WaveTrend3D(df1['close'])
    wt3d.compute_series()
    wt3d.compute_signals()

    # For quick visual inspection (if PlotWaveTrend3D had been loaded) [for Arthur's testing]
    # plot = PlotWaveTrend3D([.7, .2, .1], height=800)
    # plot.add_candles(df1, 'BTC-USDT', '1h')
    # plot.add_wavetrend(wt3d, main=True, mirror=True)
    # plot.show()

    # The actual signal
    # .astype('i4') is enough to make all True a 1 and False a 0. Also BA seems to require a list instead of array
    Signal = wt3d.divergence_bullish.astype('i4').tolist()
    print(Signal)
