# wavetrend_3d.py
# "WaveTrend3D"
# Copyright (C) 2024 Arthur Nazarian

from typing import Union
import polars as pl
import numpy as np
from scipy.ndimage import gaussian_filter1d


def cross_over(a: np.ndarray, b: Union[np.ndarray, float, int] = 0) -> np.ndarray:
    """
    Detect cross over events where `a` crosses over `b`.
    Very first element will always be False.

    Parameters
    ----------
    a : np.ndarray
        The main array to check for cross over.
    b : Union[np.ndarray, float, int], optional
        The threshold array or scalar, by default 0.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a crossover.
    """
    a = np.asarray(a)  # Ensure `a` is an array
    b = b if isinstance(b, np.ndarray) else np.full_like(a, b)

    # previous was below (or previous was equal but prior to that was below) and now is above
    cross_over_bool = (np.roll(a < b, 1) | (np.roll(a == b, 1) & np.roll(a < b, 2))) & (a > b)
    cross_over_bool[0] = False

    return cross_over_bool


def cross_down(a: np.ndarray, b: Union[np.ndarray, float, int] = 0) -> np.ndarray:
    """
    Detect cross under events where `a` crosses below `b`.
    Very first element will always be False.

    Parameters
    ----------
    a : np.ndarray
        The main array to check for cross down.
    b : Union[np.ndarray, float, int], optional
        The threshold array or scalar, by default 0.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a cross under.
    """
    a = np.asarray(a)  # Ensure `a` is an array
    b = b if isinstance(b, np.ndarray) else np.full_like(a, b)

    # previous was above (or previous was equal but prior to that was above) and now is below
    cross_under_bool = (np.roll(a > b, 1) | (np.roll(a == b, 1) & np.roll(a > b, 2))) & (a < b)
    cross_under_bool[0] = False

    return cross_under_bool


def cross(a: np.ndarray, b: Union[np.ndarray, float, int]) -> np.ndarray:
    """
    Detect crossings between two arrays or between an array and a scalar.
    Very first element will always be False.

    Parameters
    ----------
    a : np.ndarray
        The main array to check for crossings.
    b : Union[np.ndarray, float, int]
        The threshold array or scalar.

    Returns
    -------
    np.ndarray
        A boolean array where True indicates a crossing.
    """
    return cross_over(a, b) | cross_down(a, b)

def normalize_derivative(data: np.ndarray, quadratic_mean_length: Union[int, float]) -> np.ndarray:
    """
    Normalize the discrete second derivative of a data series using the quadratic mean (RMS).

    This function computes the normalized derivative of a data series. It first calculates the
    discrete second derivative, squares it, and then applies a quadratic mean (root-mean-square, RMS)
    over a specified length. The discrete second derivative is a simple approximation of the curvature or acceleration
    of the data series, giving insights into changes in trends.The result is a measure of the average magnitude
    of the second derivative, adjusted for the specified window, which is useful in signal processing to assess
    the variability or volatility of the data over time.

    Parameters
    ----------
    data : np.ndarray
        The input data series as a numpy array.
    quadratic_mean_length : Union[int, float]
        The length of the window over which to calculate the quadratic mean. This value determines
        the number of data points considered for smoothing the derivative signal.

    Returns
    -------
    np.ndarray
        An array of the same length as `data`, with the normalized derivative values. The start of the
        array is padded with NaN values to match the input size, reflecting the loss of points due
        to the calculation process. 2 elements are lost due to taking the difference of derivates
        (`x_{t} - x_{t-2}`) and another `quadratic_mean_length - 1` elements are lost due to rolling averaging

    Notes
    -----
    - The quadratic mean, or RMS, is a statistical measure of the magnitude of a varying quantity.
      It is particularly useful in signal processing as it provides a measure of the power contained
      in the signal irrespective of its direction.
    - The discrete second derivative is a simple approximation of the curvature or acceleration of
      the data series, giving insights into changes in trends.
    """

    # Calculate the discrete derivative
    derivative = data[2:] - data[:-2]

    # Calculate the quadratic mean (root-mean-square)
    squared_derivative = derivative ** 2

    # similar to moving average due to the second array being ones
    # equivalent: pd.Series(squared_derivative).rolling(quadratic_mean_length).sum().dropna().to_numpy()
    window_sum = np.convolve(squared_derivative, np.ones(quadratic_mean_length), mode='valid')

    # the actual Root Mean Squared (RMS)
    quadratic_mean = np.sqrt(window_sum / quadratic_mean_length)

    # Normalize the derivative
    normalized_derivative = derivative[quadratic_mean_length - 1:] / quadratic_mean

    # add NaN in beginning to have equal length as input
    output_length = len(data) - 2 - (quadratic_mean_length - 1)
    return np.concatenate((
        np.full(len(data) - output_length, np.nan),
        normalized_derivative
    ))


def apply_dual_pole_filter(data: np.ndarray, lookback: Union[int, float]) -> np.ndarray:

    """
        Apply a dual-pole low-pass filter to a data series for smoothing and trend analysis.

        This function implements a digital filter that averages the data while giving more
        weight to recent values, using a specific lookback period. It's designed to reduce
        high-frequency noise in the data, thereby highlighting underlying trends. The filter
        parameters are calculated to achieve a balance between responsiveness and smoothing,
        making it particularly useful in financial technical analysis for isolating major movements.

        Parameters
        ----------
        data : np.ndarray
            The input data series as a numpy array.
        lookback : Union[int, float]
            The lookback period for the filter, which influences the filter's responsiveness
            and smoothing effect. A higher value results in more smoothing.

        Returns
        -------
        np.ndarray
            The filtered data series, which emphasizes the underlying trend by reducing the
            impact of short-term fluctuations.

        Notes
        -----
        - The dual-pole filter is a form of infinite impulse response (IIR) filter that uses feedback
          to create a smoothing effect. It is characterized by its ability to provide a smoother signal
          without significant lag, which is crucial for real-time signal analysis.
        - The filter parameters (omega, alpha, beta, gamma, delta) are derived based on the lookback
          period and control the decay of influence from older data points, thereby determining the
          filter's frequency response characteristics.
        - There are different versions of dual pole filters. This is a direct port from Justin Dehorty's
          TradingView implementation
        """
    omega = -99 * np.pi / (70 * lookback)
    alpha = np.exp(omega)
    beta = -np.power(alpha, 2)
    gamma = np.cos(omega) * 2 * alpha
    delta = 1 - gamma - beta

    # moving average with window=2 and 'backward-fill' as first element to make the shape equal to `data`
    sliding_avg = np.insert((data[:-1] + data[1:]) / 2, 0, data[0])

    # compute first two elements
    filtered = np.empty_like(data)

    # Compute filtered values; a bit slow in Python due to the loop
    for i in range(2, len(data)):
        if np.isnan(sliding_avg[i]):
            filtered[i] = np.nan
        else:
            filtered[i] = (
                delta * sliding_avg[i]
                + gamma * (filtered[i - 1] if not np.isnan(filtered[i - 1]) else 0)
                + beta  * (filtered[i - 2] if not np.isnan(filtered[i - 2]) else 0)
            )
    return filtered


def get_oscillator(src, smoothing_frequency: float, quadratic_mean_length: Union[int, float]):
    """
    Generate an oscillator signal from a data series using normalization and filtering techniques.

    This function first normalizes the derivative of the input data using a quadratic mean over a
    specified length, then applies a hyperbolic tangent to ensure the values are bounded between -1
    and 1. The result is then filtered using a dual-pole low-pass filter to reduce noise and
    highlight longer-term trends or cycles. The output is a smoothed oscillator signal that reflects
    the underlying momentum or trend strength of the input data, suitable for technical analysis
    in trading systems.

    Parameters
    ----------
    src : np.ndarray
        The input data series as a numpy array.
    smoothing_frequency : float
        The frequency parameter for the dual-pole filter, controlling the level of smoothing.
    quadratic_mean_length : Union[int, float]
        The length of the window over which to calculate the quadratic mean for normalization.

    Returns
    -------
    np.ndarray
        A smoothed oscillator signal derived from the input data, intended for identifying
        trend strength and potential reversal points in financial markets.
    """
    tanh_norm_deriv = np.tanh(normalize_derivative(src, quadratic_mean_length))
    # return np.insert(apply_dual_pole_filter(tanh_norm_deriv, smoothing_frequency), 0, np.full(2, np.nan))
    return apply_dual_pole_filter(tanh_norm_deriv, smoothing_frequency)


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
