"""Williams %R.

    %R = (highest_high_n − close) / (highest_high_n − lowest_low_n) × (−100)

Values range from -100 to 0.  Readings near -100 indicate oversold conditions;
readings near 0 indicate overbought conditions.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
Williams, L. (1979). How I Made One Million Dollars Last Year Trading
Commodities.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.utils.rolling import rolling_max, rolling_min


@pandas_aware
def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Calculate Williams %R.

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    close:
        1-D array-like of closing prices.
    period:
        Look-back window (default 14).

    Returns
    -------
    np.ndarray
        Williams %R values in the range ``[-100, 0]``; ``NaN`` for the first
        ``period - 1`` entries.
    """
    high = to_float_array(high)
    low = to_float_array(low)
    close = to_float_array(close)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have the same length")

    hh = rolling_max(high, period)
    ll = rolling_min(low, period)

    denom = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = np.where(denom != 0, (hh - close) / denom * -100.0, np.nan)

    return wr
