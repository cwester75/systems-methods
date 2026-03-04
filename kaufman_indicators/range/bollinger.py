"""Bollinger Bands.

    Middle Band = SMA(close, period)
    Upper Band  = Middle Band + k × rolling_std(close, period)
    Lower Band  = Middle Band − k × rolling_std(close, period)

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 20.
Bollinger, J. (2001). *Bollinger on Bollinger Bands*.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.rolling import rolling_mean, rolling_std


class BollingerResult(NamedTuple):
    """Container returned by :func:`bollinger_bands`."""

    middle: np.ndarray
    """Middle band (SMA)."""
    upper: np.ndarray
    """Upper band."""
    lower: np.ndarray
    """Lower band."""
    bandwidth: np.ndarray
    """(upper − lower) / middle × 100 – width as a percentage of the middle."""
    percent_b: np.ndarray
    """(close − lower) / (upper − lower) – position within the bands."""


def bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> BollingerResult:
    """Calculate Bollinger Bands.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        SMA and rolling-std window (default 20).
    num_std:
        Number of standard deviations for the band width (default 2.0).

    Returns
    -------
    BollingerResult
        Named tuple with ``middle``, ``upper``, ``lower``, ``bandwidth``,
        and ``percent_b`` arrays, each the same length as *prices*; ``NaN``
        where data is insufficient.
    """
    prices = to_float_array(prices)

    middle = rolling_mean(prices, period)
    std = rolling_std(prices, period)

    upper = middle + num_std * std
    lower = middle - num_std * std

    with np.errstate(invalid="ignore", divide="ignore"):
        bandwidth = np.where(middle != 0, (upper - lower) / middle * 100.0, np.nan)
        band_range = upper - lower
        percent_b = np.where(band_range != 0, (prices - lower) / band_range, np.nan)

    return BollingerResult(middle, upper, lower, bandwidth, percent_b)
