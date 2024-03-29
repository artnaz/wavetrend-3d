# signal_processing.py
# implements signal processing methods the classical "WaveTrend" as an oscillator for financial data series
# Copyright (C) 2024 Arthur Nazarian

from typing import Union
import numpy as np


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

    # TODO: Vectorized version below doesn't give correct results
    # valid_start = np.where(~np.isnan(data))[0][0]
    #
    # filtered_shift1 = np.nan_to_num(np.insert(filtered[1:], 0, 0))
    # filtered_shift2 = np.nan_to_num(np.insert(filtered[2:], 0, [0, 0]))
    #
    # filtered[valid_start:] = (
    #     delta * sliding_avg[valid_start:]
    #     + gamma * filtered_shift1[valid_start:]
    #     + beta * filtered_shift2[valid_start:]
    # )
    # return filtered


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
