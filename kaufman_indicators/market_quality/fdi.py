"""Fractal Dimension Index (FDI).

The FDI estimates the fractal dimension of a price series over a rolling
window.  Values near 1.5 indicate a random walk; values closer to 1.0
indicate a trending market; values closer to 2.0 indicate a mean-reverting
or noisy market.

The implementation follows Sevcik's simplified approach:

    D = [log(N) − log(N−1)] / log(2)

where N is the number of "self-similar pieces" estimated from the price path
length.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 23.
Sevcik, C. (1990). A procedure to estimate the fractal dimension of waveforms.
*Complexity International*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware


@pandas_aware
def fdi(prices: np.ndarray, period: int = 30) -> np.ndarray:
    """Calculate the Fractal Dimension Index over a rolling *period*.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Rolling window size (default 30).

    Returns
    -------
    np.ndarray
        FDI values; same length as *prices*; ``NaN`` for the first
        ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n < period:
        return result

    for i in range(period - 1, n):
        window = prices[i - period + 1: i + 1]
        hi = np.max(window)
        lo = np.min(window)
        price_range = hi - lo

        if price_range == 0:
            result[i] = np.nan
            continue

        # Path length normalized by price range
        path = np.sum(np.abs(np.diff(window))) / price_range
        # Fractal dimension estimate
        if path > 0:
            result[i] = 1.0 + (np.log(path) + np.log(2)) / np.log(2 * (period - 1))
        else:
            result[i] = np.nan

    return result
