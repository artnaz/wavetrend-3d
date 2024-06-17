# wavetrend_3d.py
# Defines the "WaveTrend3D" class
# Copyright (C) 2024 Arthur Nazarian

from typing import Union
import polars as pl
import numpy as np
from scipy.ndimage import gaussian_filter1d

from wavetrend_3d.signal_processing import get_oscillator
from wavetrend_3d.common.signal_tools import cross_over, cross_down


class WaveTrend3D:
    def __init__(self, data: Union[np.ndarray, pl.Series]):
        self._data = None
        self.data = data  # Uses property setter defined below

        # outputs - series
        self.series_fast: np.ndarray = None
        self.series_norm: np.ndarray = None
        self.series_slow: np.ndarray = None

        # outputs - signals
        self.fast_norm_cross_bullish: np.ndarray = None
        self.fast_norm_cross_bearish: np.ndarray = None
        self.divergence_bullish: np.ndarray = None
        self.divergence_bearish: np.ndarray = None

        self.fast_median_cross_bullish: np.ndarray = None
        self.norm_median_cross_bullish: np.ndarray = None
        self.slow_median_cross_bullish: np.ndarray = None
        self.fast_median_cross_bearish: np.ndarray = None
        self.norm_median_cross_bearish: np.ndarray = None
        self.slow_median_cross_bearish: np.ndarray = None

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

    def compute_series(
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
    ) -> None:
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
        self.series_fast = f_length * signal_fast

        n_lookback = n_smoothing * oscillator_lookback
        signal_normal = get_oscillator(self.data, n_lookback, quadratic_mean_length)
        self.series_norm = n_length * signal_normal

        s_lookback = s_smoothing * oscillator_lookback
        signal_slow = get_oscillator(self.data, s_lookback, quadratic_mean_length)
        self.series_slow = s_length * signal_slow

    def get_series(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the fast, normal and slow series that were computed in `compute_series()` as a tuple.
        If the method was not run previously, then it will be done in this `get_series()` method.
        Keyword arguments will be passed.
        """
        if self.series_fast is None:
            self.compute_series(**kwargs)
        return self.series_fast, self.series_norm, self.series_slow

    @staticmethod
    def _cross_two_series(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns bullish, bearish"""
        bull = cross_over(a, b) & (b < 0)
        bear = cross_down(a, b) & (b > 0)
        return bull, bear

    def _compute_fast_norm_cross(self):
        self.fast_norm_cross_bullish, self.fast_norm_cross_bearish = \
            self._cross_two_series(self.series_fast, self.series_norm)

    def _compute_median_crosses(self):
        self.fast_median_cross_bearish = cross_down(self.series_fast, 0)
        self.norm_median_cross_bearish = cross_down(self.series_norm, 0)
        self.slow_median_cross_bearish = cross_down(self.series_slow, 0)

        self.fast_median_cross_bullish = cross_over(self.series_fast, 0)
        self.norm_median_cross_bullish = cross_over(self.series_norm, 0)
        self.slow_median_cross_bullish = cross_over(self.series_slow, 0)

    @staticmethod
    def _divergence_three_series(norm: np.ndarray, slow: np.ndarray, crosses: np.ndarray, direction: str,
                                 lookback_bound: int, proportional_decrease_bound: float, gaussian_sigma: float = 6.):
        # (1) Get distance to previous cross
        distance_to_prev = np.insert(np.diff(np.where(crosses)[0]), 0, -9999)

        # (2) Keep only the values of the crosses and previous crosses. Skip first cross as that has no prior reference.
        cross_values = norm[crosses]
        cross_values_prev = np.insert(cross_values[:-1], 0, np.nan)
        proportion = cross_values / cross_values_prev

        # (3) Apply Gaussian filter to the slowest series to help determining longer term trend
        series_smooth = gaussian_filter1d(slow, sigma=gaussian_sigma)

        # Conditions checks: (1) previous cross within lookback threshold; (2) proportional size smaller than threshold;
        # (3) slow series confirms tren
        match direction:
            case 'bull':
                # This is a sparse array denoting which of the crosses (subset of timeseries) had divergence
                divergence = (distance_to_prev <= lookback_bound) & \
                    (proportion <= proportional_decrease_bound) & \
                    (slow[crosses] >= series_smooth[crosses])
            case 'bear':
                divergence = (distance_to_prev <= lookback_bound) & \
                    (proportion <= proportional_decrease_bound) & \
                    (slow[crosses] <= series_smooth[crosses])
            case _:
                raise ValueError('`direction` should be either "bull" or "bear".')

        # Create full-length array
        indices = np.where(crosses)[0][divergence]
        out = np.full_like(norm, False, 'bool')
        out[indices] = True
        return out

    def _compute_divergence(self, lookback_bound: int, proportional_decrease_bound: float):
        if self.fast_norm_cross_bullish is None:
            self._compute_fast_norm_cross()

        kwargs = dict(norm=self.series_norm, slow=self.series_slow, lookback_bound=lookback_bound,
                      proportional_decrease_bound=proportional_decrease_bound)
        self.divergence_bullish = self._divergence_three_series(
            **kwargs, crosses=self.fast_norm_cross_bullish, direction='bull')
        self.divergence_bearish = self._divergence_three_series(
            **kwargs, crosses=self.fast_norm_cross_bearish, direction='bear')

    def compute_signals(self, lookback_bound=40, proportional_decrease_bound=0.4):
        if self.series_norm is None:
            raise ProcessLookupError(
                'No WaveTrend series exist to compute signals. Run `compute_series()` method first.')
        self._compute_median_crosses()
        self._compute_fast_norm_cross()
        self._compute_divergence(lookback_bound=lookback_bound, proportional_decrease_bound=proportional_decrease_bound)
