"""Stochastic Oscillator.

    %K = (close − lowest_low_n) / (highest_high_n − lowest_low_n) × 100
    %D = SMA(%K, d_period)

Values near 100 indicate that the close is near the top of the recent range;
values near 0 indicate a close near the bottom.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
Lane, G. C. (1984). Lane's Stochastics. *Technical Analysis of Stocks &
Commodities*.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.rolling import rolling_max, rolling_min, rolling_mean


class StochasticResult(NamedTuple):
    """Container returned by :func:`stochastic`."""

    k: np.ndarray
    """%K – fast stochastic."""
    d: np.ndarray
    """%D – smoothed stochastic (signal line)."""


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> StochasticResult:
    """Calculate the Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    close:
        1-D array-like of closing prices.
    k_period:
        Look-back period for %K (default 14).
    d_period:
        Smoothing period for %D (default 3).

    Returns
    -------
    StochasticResult
        Named tuple with ``k`` and ``d`` arrays, each the same length as the
        inputs; ``NaN`` where data is insufficient.
    """
    high = to_float_array(high)
    low = to_float_array(low)
    close = to_float_array(close)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have the same length")

    highest_high = rolling_max(high, k_period)
    lowest_low = rolling_min(low, k_period)

    denom = highest_high - lowest_low
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(denom != 0, (close - lowest_low) / denom * 100.0, np.nan)

    d = rolling_mean(k, d_period)
    return StochasticResult(k, d)
