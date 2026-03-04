"""Realized (Historical) Volatility.

Realized volatility is computed as the annualized standard deviation of
log returns over a rolling window.

    rv_t = std(log_returns_{t-period+1..t}) × sqrt(annualization_factor)

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 21.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array, log_returns
from kaufman_indicators.utils.rolling import rolling_std


def realized_vol(
    prices: np.ndarray,
    period: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Rolling realized (historical) volatility.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Rolling window for the standard deviation of log returns (default 20).
    annualize:
        If ``True`` (default), multiply by ``sqrt(periods_per_year)``.
    periods_per_year:
        Trading periods in a year used for annualization (default 252).

    Returns
    -------
    np.ndarray
        Realized volatility values; same length as *prices*; ``NaN`` for the
        first *period* entries (need at least *period* log returns).
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    lr = log_returns(prices)               # length n-1
    rv = rolling_std(lr, period)           # length n-1, NaN for first period-1

    if annualize:
        rv = rv * np.sqrt(periods_per_year)

    # rv[i] corresponds to prices[i+1]
    result[period:] = rv[period - 1:]
    return result
