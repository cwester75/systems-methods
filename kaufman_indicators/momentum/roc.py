"""Rate of Change (ROC) momentum indicator.

    ROC = (price_t − price_{t−n}) / price_{t−n} × 100

A positive ROC indicates upward momentum; negative indicates downward.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array


def roc(prices: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Look-back period (default 12).

    Returns
    -------
    np.ndarray
        ROC values as percentages; ``NaN`` for the first *period* entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    prev = prices[:-period]
    with np.errstate(invalid="ignore", divide="ignore"):
        result[period:] = np.where(
            prev != 0,
            (prices[period:] - prev) / prev * 100.0,
            np.nan,
        )
    return result
