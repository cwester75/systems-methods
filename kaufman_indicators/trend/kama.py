"""Kaufman Adaptive Moving Average (KAMA).

KAMA adapts its speed to market noise by using the Efficiency Ratio to
scale between a fast and a slow EMA smoothing constant.

    SC     = (ER * (fast_sc − slow_sc) + slow_sc) ** 2
    KAMA_t = KAMA_{t-1} + SC * (price_t − KAMA_{t-1})

where
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 17.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.trend.efficiency_ratio import efficiency_ratio


def kama(
    prices: np.ndarray,
    period: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> np.ndarray:
    """Calculate the Kaufman Adaptive Moving Average.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Efficiency Ratio look-back window (default 10).
    fast:
        Fast EMA period (default 2).
    slow:
        Slow EMA period (default 30).

    Returns
    -------
    np.ndarray
        KAMA values, same length as *prices*; ``NaN`` for the first
        *period* positions.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    er = efficiency_ratio(prices, period)

    # Seed KAMA at the first valid ER position
    seed_idx = period
    result[seed_idx] = prices[seed_idx]

    for i in range(seed_idx + 1, n):
        er_i = er[i]
        if np.isnan(er_i):
            er_i = 0.0
        sc = (er_i * (fast_sc - slow_sc) + slow_sc) ** 2
        result[i] = result[i - 1] + sc * (prices[i] - result[i - 1])

    return result
