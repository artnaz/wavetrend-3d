# signal_tools.py
# Functions that help generate signals
# Copyright (C) 2024 Arthur Nazarian

import numpy as np
from typing import Union


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


# #  Slower than Numpy due to relatively simple function and the overhead of conversions, aliases, etc.
# import polars as pl
# def _cross_polars(a: Union[np.ndarray, pl.Series, float, int], b: Union[np.ndarray, pl.Series, float, int]):
#     df = pl.LazyFrame({'input': a, 'threshold': b})
#     df = (
#         df
#         .with_columns(
#             pl.col('input').shift(1).alias('input-1'),
#             pl.col('threshold').shift(1).alias('threshold-1'),
#             pl.col('input').shift(2).alias('input-2'),
#             pl.col('threshold').shift(2).alias('threshold-2'),
#         )
#         .with_columns(
#             cross_over=
#             (pl.col('input-1').lt(pl.col('threshold-1')) |
#              (pl.col('input-1').eq(pl.col('threshold-1')) & pl.col('input-2').lt(pl.col('threshold-2')))) &
#             pl.col('input').gt(pl.col('threshold')),
#             cross_down=
#             (pl.col('input-1').gt(pl.col('threshold-1')) |
#              (pl.col('input-1').eq(pl.col('threshold-1')) & pl.col('input-2').gt(pl.col('threshold-2')))) &
#             pl.col('input').lt(pl.col('threshold'))
#         )
#         .with_columns(
#             cross=pl.col('cross_over') | pl.col('cross_down')
#         )
#     )
#     # would need to select cross_over/cross_down/cross
#     return df.select(pl.exclude('input', 'threshold')).fill_null(False).collect().to_series()
