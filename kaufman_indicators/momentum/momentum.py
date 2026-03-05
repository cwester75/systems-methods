"""Momentum indicator.

    Momentum = price_t − price_{t−n}

The simplest measure of the speed of price movement.  A positive value
indicates the price is higher than *n* bars ago; negative indicates it is
lower.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array


def momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Momentum (absolute price change over *period* bars).

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Look-back period (default 10).

    Returns
    -------
    np.ndarray
        Momentum values; ``NaN`` for the first *period* entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    result[period:] = prices[period:] - prices[:-period]
    return result
