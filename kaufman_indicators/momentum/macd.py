"""Moving Average Convergence/Divergence (MACD).

    MACD line  = EMA(prices, fast) − EMA(prices, slow)
    Signal     = EMA(MACD line, signal_period)
    Histogram  = MACD line − Signal

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 12.
Appel, G. (1979). *The Moving Average Convergence-Divergence Trading Method*.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.trend.moving_averages import ema


class MACDResult(NamedTuple):
    """Container returned by :func:`macd`."""

    macd_line: np.ndarray
    """MACD line = fast EMA − slow EMA."""
    signal: np.ndarray
    """Signal line = EMA of the MACD line."""
    histogram: np.ndarray
    """Histogram = MACD line − signal line."""


@pandas_aware
def macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> MACDResult:
    """Calculate MACD, signal line, and histogram.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    fast:
        Fast EMA period (default 12).
    slow:
        Slow EMA period (default 26).
    signal_period:
        Signal EMA period (default 9).

    Returns
    -------
    MACDResult
        Named tuple with ``macd_line``, ``signal``, and ``histogram`` arrays,
        each the same length as *prices*; ``NaN`` where data is insufficient.
    """
    prices = to_float_array(prices)

    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return MACDResult(macd_line, signal_line, histogram)
