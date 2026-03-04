"""Donchian Channels.

The Donchian Channel draws a band between the highest high and the lowest low
over a rolling *period* window.  A mid-line (average of the two) is also
provided.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 5.
Donchian, R. D. (1960). High Finance in Copper. *Financial Analysts Journal*.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.rolling import rolling_max, rolling_min


class DonchianResult(NamedTuple):
    """Container returned by :func:`donchian_channels`."""

    upper: np.ndarray
    """Rolling highest high."""
    lower: np.ndarray
    """Rolling lowest low."""
    mid: np.ndarray
    """Mid-line = (upper + lower) / 2."""


def donchian_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
) -> DonchianResult:
    """Calculate Donchian Channels.

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    period:
        Look-back window (default 20).

    Returns
    -------
    DonchianResult
        Named tuple with ``upper``, ``lower``, and ``mid`` arrays, each the
        same length as the inputs; ``NaN`` for the first ``period - 1``
        entries.
    """
    high = to_float_array(high)
    low = to_float_array(low)

    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    upper = rolling_max(high, period)
    lower = rolling_min(low, period)
    mid = (upper + lower) / 2.0

    return DonchianResult(upper, lower, mid)
