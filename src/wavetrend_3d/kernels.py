import numpy as np

def gaussian_kernel_smoothing(series, lookback, start_at_bar):
    """
    Apply a Gaussian kernel smoothing to the input series.

    Parameters
    ----------
    series : np.ndarray
        The source series.
    lookback : int
        The number of bars used for the estimation.
    start_at_bar : int
        Bar index on which to start regression.

    Returns
    -------
    np.ndarray
        The smoothed series.
    """
    n = len(series)
    smoothed_series = np.zeros_like(series, dtype=float)

    for t in range(start_at_bar, n):
        current_weight = 0.0
        cumulative_weight = 0.0

        for i in range(t + 1):
            y = series[i]
            w = np.exp(-np.power(t - i, 2) / (2 * np.power(lookback, 2)))
            current_weight += y * w
            cumulative_weight += w

        smoothed_series[t] = current_weight / cumulative_weight if cumulative_weight != 0 else 0

    return smoothed_series

# Example usage:
series = np.random.rand(100)
lookback = 6
start_at_bar = 0

smoothed_series = gaussian_kernel_smoothing(series, lookback, start_at_bar)
print(smoothed_series)