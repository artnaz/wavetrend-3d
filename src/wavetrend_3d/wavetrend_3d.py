# wavetrend_3d.py
# Defines the "WaveTrend3D" class
# Copyright (C) 2024 Arthur Nazarian

from typing import Union, Tuple
import polars as pl
import numpy as np

from src.wavetrend_3d.signal_processing import get_oscillator


class WaveTrend3D:
    def __init__(self, data: Union[np.ndarray, pl.Series]):
        self._data = None
        self.data = data  # Uses property setter defined below

    @property
    def data(self) -> np.ndarray:
        """Property getter for `data`"""
        return self._data

    @data.setter
    def data(self, value: Union[np.ndarray, pl.Series]):
        """Property setter for `data`, with validation and conversion"""
        if not isinstance(value, (np.ndarray, pl.Series)):
            raise ValueError('`data` should be passed as a Numpy array or Polars Series')
        self._data = value.to_numpy() if isinstance(value, pl.Series) else value.flatten()

    def get_series(
        self,
        oscillator_lookback: int = 20,
        quadratic_mean_length: int = 50,
        f_length: float = 0.75,
        n_length: float = 1.00,
        s_length: float = 1.75,
        f_smoothing: float = 0.45,
        n_smoothing: float = 1.00,
        s_smoothing: float = 2.50,
        cog_length: int = None,
        ma_length: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a three series of oscillator signals at different frequencies and scales from a data series.

        This function wraps the oscillator generation process, applying it to three different frequency settings
        (fast, normal, slow) to analyze the data at multiple scales, possibly representing multiple timeframes.
        Each oscillator is scaled and offset to provide distinct insights into the market's momentum and
        potential reversal points.

        Parameters
        ----------
        oscillator_lookback : int, optional
            The number of bars to use for signal smoothing, default is 20.
        quadratic_mean_length : int, optional
            The window length for calculating the quadratic mean for normalization, default is 50.
        f_length : float, optional
            The length scale factor for the fast oscillator, default is 0.75.
        n_length : float, optional
            The length scale factor for the normal oscillator, default is 1.00.
        s_length : float, optional
            The length scale factor for the slow oscillator, default is 1.75.
        f_smoothing : float, optional
            The smoothing scale factor for the fast frequency, default is 0.45.
        n_smoothing : float, optional
            The smoothing scale factor for the normal frequency, default is 1.00.
        s_smoothing : float, optional
            The smoothing scale factor for the slow frequency, default is 2.50.
        cog_length : int, optional
            The length for calculating the center of gravity of the price distribution, applied to the input series.
            Default is "None", thus no transformation is applied.
        ma_length : int, optional
            The window length for calculating the exponential moving average of the price distribution,
            applied to the input series. Default is "None", thus no transformation is applied.

        Returns
        -------
        tuple
        - A tuple of three numpy arrays, each representing a different oscillator signal (fast, normal, slow) derived
          from the input data. These signals provide a multi-scale perspective on the market's momentum and
          trend strength.

        Notes
        -----
        - Visual additions, in particular emphasising a certain oscillator signal, that are in the original TradingView
          script are currently not implemented.
        - Default parameters are the same as in the original TradingView script.

        """
        if cog_length is not None:
            raise NotImplementedError('Transforming the input series with the Centre of Gravity (CoG) is '
                                      'currently not supported')
        if ma_length is not None:
            raise NotImplementedError('Transforming the input series with an Exponential Moving Average (EMA) is '
                                      'currently not supported')

        f_lookback = f_smoothing * oscillator_lookback
        signal_fast = get_oscillator(self.data, f_lookback, quadratic_mean_length)
        series_fast = f_length * signal_fast

        n_lookback = n_smoothing * oscillator_lookback
        signal_normal = get_oscillator(self.data, n_lookback, quadratic_mean_length)
        series_norm = n_length * signal_normal

        s_lookback = s_smoothing * oscillator_lookback
        signal_slow = get_oscillator(self.data, s_lookback, quadratic_mean_length)
        series_slow = s_length * signal_slow

        return series_fast, series_norm, series_slow
