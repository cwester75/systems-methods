"""Kaufman Efficiency Ratio (ER).

The Efficiency Ratio measures how directionally efficient a price move is
relative to its total path length over a given look-back period.

    ER = |net_change| / sum_of_absolute_daily_changes

A value close to 1 indicates a strongly trending market; a value close to 0
indicates a noisy, sideways market.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 17.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.rolling import rolling_sum


def efficiency_ratio(prices: np.ndarray, period: int = 10) -> np.ndarray:
    """Calculate the Efficiency Ratio over *period* bars.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Look-back window (default 10).

    Returns
    -------
    np.ndarray
        ER values, same length as *prices*; ``NaN`` for the first
        ``period`` positions (need at least *period* price changes).
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    # Absolute daily changes
    abs_changes = np.abs(np.diff(prices))               # length n-1
    # Rolling sum of abs_changes over *period* bars
    path = rolling_sum(abs_changes, period)             # length n-1, NaN for first period-1
    # Net change over *period* bars (comparing price to price *period* bars ago)
    net = np.abs(prices[period:] - prices[:-period])    # length n-period

    # Align: path[period-1:] corresponds to net[0:]
    path_aligned = path[period - 1:]                    # length n-period
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(path_aligned != 0, net / path_aligned, 0.0)

    result[period:] = er
    return result
