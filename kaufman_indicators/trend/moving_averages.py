"""Common moving averages.

Implements:
- SMA  – Simple Moving Average
- EMA  – Exponential Moving Average
- WMA  – Weighted Moving Average (linearly weighted)
- DEMA – Double Exponential Moving Average
- TEMA – Triple Exponential Moving Average

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 7.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.utils.rolling import rolling_mean


@pandas_aware
def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    Parameters
    ----------
    prices:
        1-D array-like of values.
    period:
        Number of bars to average.

    Returns
    -------
    np.ndarray
        SMA values; ``NaN`` for the first ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    return rolling_mean(prices, period)


@pandas_aware
def ema(prices: np.ndarray, period: int, alpha: float | None = None) -> np.ndarray:
    """Exponential Moving Average.

    Parameters
    ----------
    prices:
        1-D array-like of values.
    period:
        Number of bars (used to compute ``alpha = 2 / (period + 1)`` when
        *alpha* is ``None``).
    alpha:
        Custom smoothing factor.  If supplied, *period* is still used to
        determine when to seed the first EMA value (from the mean of the
        first *period* bars).

    Returns
    -------
    np.ndarray
        EMA values; ``NaN`` for the first ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if alpha is None:
        alpha = 2.0 / (period + 1)

    # Find the first index where we have *period* consecutive non-NaN values
    first_valid = np.argmax(~np.isnan(prices))
    available = n - first_valid
    if available < period:
        return result

    seed_idx = first_valid + period - 1
    result[seed_idx] = np.mean(prices[first_valid: first_valid + period])
    for i in range(seed_idx + 1, n):
        if np.isnan(prices[i]):
            continue
        result[i] = result[i - 1] + alpha * (prices[i] - result[i - 1])
    return result


@pandas_aware
def wma(prices: np.ndarray, period: int) -> np.ndarray:
    """Linearly Weighted Moving Average.

    Weights are 1, 2, …, *period* (most recent bar gets weight *period*).

    Parameters
    ----------
    prices:
        1-D array-like of values.
    period:
        Number of bars.

    Returns
    -------
    np.ndarray
        WMA values; ``NaN`` for the first ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()

    for i in range(period - 1, n):
        result[i] = np.dot(prices[i - period + 1: i + 1], weights) / weight_sum
    return result


@pandas_aware
def dema(prices: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average.

    DEMA = 2 * EMA(prices, period) − EMA(EMA(prices, period), period)

    Reduces lag compared to a standard EMA.

    Parameters
    ----------
    prices:
        1-D array-like of values.
    period:
        EMA period.

    Returns
    -------
    np.ndarray
        DEMA values.
    """
    e1 = ema(prices, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2


@pandas_aware
def tema(prices: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average.

    TEMA = 3 * EMA1 − 3 * EMA2 + EMA3

    where EMA1 = EMA(prices), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2).

    Parameters
    ----------
    prices:
        1-D array-like of values.
    period:
        EMA period.

    Returns
    -------
    np.ndarray
        TEMA values.
    """
    e1 = ema(prices, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * e1 - 3.0 * e2 + e3
