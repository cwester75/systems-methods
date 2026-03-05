"""Price Z-Score.

    Z = (price − SMA(price, period)) / rolling_std(price, period)

Measures how many standard deviations the current price is from its rolling
mean.  Positive values indicate price is above the mean; negative values
indicate it is below.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 20.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.rolling import rolling_mean, rolling_std


def price_zscore(
    prices: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Rolling Price Z-Score.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Rolling window for the mean and standard deviation (default 20).

    Returns
    -------
    np.ndarray
        Z-Score values; ``NaN`` for the first ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n < period:
        return result

    mean = rolling_mean(prices, period)
    std = rolling_std(prices, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(std != 0, (prices - mean) / std, 0.0)

    # Restore NaN for warmup period
    result[:period - 1] = np.nan

    return result
