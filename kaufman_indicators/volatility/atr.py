"""Average True Range (ATR).

The ATR smooths the True Range using Wilder's smoothing method to produce an
average measure of market volatility.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 21.
Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array, wilders_smooth
from kaufman_indicators.volatility.true_range import true_range as compute_tr


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calculate the Average True Range using Wilder's smoothing.

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    close:
        1-D array-like of closing prices.
    period:
        Smoothing period (default 14).

    Returns
    -------
    np.ndarray
        ATR values, same length as inputs; ``NaN`` for the first
        ``period - 1`` entries.
    """
    high = to_float_array(high)
    low = to_float_array(low)
    close = to_float_array(close)

    tr = compute_tr(high, low, close)
    return wilders_smooth(tr, period)
