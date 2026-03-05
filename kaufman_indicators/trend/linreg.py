"""Linear Regression indicators.

Provides rolling ordinary-least-squares linear regression with helpers for
the regression line value (fitted value at each end-point), slope, intercept,
and a *look-ahead* forecast.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 5.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from kaufman_indicators.utils.math_helpers import to_float_array


class LinRegResult(NamedTuple):
    """Container returned by :func:`linreg`."""

    value: np.ndarray
    """Fitted value at the last bar of each rolling window (same as the
    forecast for ``offset=0``)."""
    slope: np.ndarray
    """Slope of the regression line in price-per-bar units."""
    intercept: np.ndarray
    """Y-intercept of the regression line."""
    r_squared: np.ndarray
    """Coefficient of determination (R²) of the regression."""


def linreg(prices: np.ndarray, period: int = 14) -> LinRegResult:
    """Rolling linear regression over *period* bars.

    For each bar ``t`` (with ``t >= period - 1``) fits a line through the
    ``period`` most-recent closing prices against an integer time axis
    ``[0, 1, …, period - 1]`` and records the *end-point* value (the fitted
    value at ``x = period - 1``).

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Look-back window (default 14).

    Returns
    -------
    LinRegResult
        Named tuple with fields ``value``, ``slope``, and ``intercept``,
        each a 1-D array the same length as *prices*; ``NaN`` for the first
        ``period - 1`` entries.
    """
    prices = to_float_array(prices)
    n = len(prices)

    value = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    intercept = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)

    if n < period:
        return LinRegResult(value, slope, intercept, r_squared)

    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = prices[i - period + 1: i + 1]
        y_mean = y.mean()
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_yy = np.sum((y - y_mean) ** 2)
        s = ss_xy / ss_xx
        b = y_mean - s * x_mean
        value[i] = s * (period - 1) + b
        slope[i] = s
        intercept[i] = b
        if ss_yy != 0:
            r_squared[i] = (ss_xy ** 2) / (ss_xx * ss_yy)
        else:
            r_squared[i] = 1.0  # all y-values identical → perfect fit

    return LinRegResult(value, slope, intercept, r_squared)


def linreg_forecast(prices: np.ndarray, period: int = 14, offset: int = 1) -> np.ndarray:
    """Rolling linear regression *forecast* ``offset`` bars ahead.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Look-back window (default 14).
    offset:
        Number of bars ahead to forecast (default 1).

    Returns
    -------
    np.ndarray
        Forecast values; ``NaN`` for the first ``period - 1`` entries.
    """
    result = linreg(prices, period)
    return result.value + result.slope * offset
