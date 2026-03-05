"""Relative Strength Index (RSI).

The RSI compares the magnitude of recent gains to recent losses to measure
overbought and oversold conditions.

    RS  = avg_gain / avg_loss
    RSI = 100 − 100 / (1 + RS)

Wilder's smoothing (alpha = 1/period) is used for the averages.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware


@pandas_aware
def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder's smoothing.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Smoothing period (default 14).

    Returns
    -------
    np.ndarray
        RSI values in the range ``[0, 100]``; ``NaN`` for the first *period*
        entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed the first average with the simple mean of the first *period* values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    alpha = 1.0 / period

    # First RSI value at index *period*
    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, n - 1):
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha
        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result
