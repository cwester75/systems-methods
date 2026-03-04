"""Common mathematical helpers shared across indicator modules."""

import numpy as np


def to_float_array(x) -> np.ndarray:
    """Convert *x* to a 1-D float64 numpy array."""
    return np.asarray(x, dtype=float).ravel()


def ema_weights(n: int, alpha: float | None = None) -> np.ndarray:
    """Return EMA decay weights for *n* periods.

    Useful for weighted convolution or cross-validation calculations where
    explicit EMA weights are required rather than the recursive formula.

    Parameters
    ----------
    n:
        Number of periods.
    alpha:
        Smoothing factor.  Defaults to ``2 / (n + 1)`` when ``None``.

    Returns
    -------
    np.ndarray
        Array of length *n* with the most-recent weight at index 0.
    """
    if alpha is None:
        alpha = 2.0 / (n + 1)
    powers = np.arange(n, dtype=float)
    weights = (1.0 - alpha) ** powers
    return weights / weights.sum()


def wilders_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Apply Wilder's smoothing (used in ATR and RSI calculations).

    The first smoothed value is the simple mean of the first *period*
    values.  Subsequent values use ``alpha = 1 / period``.

    Parameters
    ----------
    values:
        1-D array of input values.
    period:
        Smoothing period.

    Returns
    -------
    np.ndarray
        Smoothed series, same length as *values*; ``NaN`` for the first
        ``period - 1`` entries.
    """
    values = to_float_array(values)
    n = len(values)
    result = np.full(n, np.nan)
    if n < period:
        return result

    alpha = 1.0 / period
    result[period - 1] = np.mean(values[:period])
    for i in range(period, n):
        result[i] = result[i - 1] * (1.0 - alpha) + values[i] * alpha
    return result


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: ``ln(p_t / p_{t-1})``.

    Returns an array one element shorter than *prices*, with no ``NaN``
    padding.
    """
    prices = to_float_array(prices)
    return np.log(prices[1:] / prices[:-1])


def sign(x: float) -> int:
    """Return the arithmetic sign of *x* (+1, -1, or 0).

    Provided as a helper for indicator logic that needs to track sign
    changes in price differences or oscillator values.
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
