"""True Range (TR).

The True Range is the greatest of:
- Current High − Current Low
- |Current High − Previous Close|
- |Current Low  − Previous Close|

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 21.
Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware


@pandas_aware
def true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Calculate the True Range.

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    close:
        1-D array-like of closing prices.

    Returns
    -------
    np.ndarray
        True Range values, same length as inputs; the first element is
        simply ``high[0] - low[0]`` (no previous close is available).
    """
    high = to_float_array(high)
    low = to_float_array(low)
    close = to_float_array(close)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have the same length")

    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    prev_close = close[:-1]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close)
    lc = np.abs(low[1:] - prev_close)

    tr[1:] = np.maximum(hl, np.maximum(hc, lc))
    return tr
