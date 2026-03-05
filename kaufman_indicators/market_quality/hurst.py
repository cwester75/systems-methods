"""Hurst Exponent.

The Hurst Exponent (H) characterises the long-term memory of a time series:

- H > 0.5  – persistent / trending behaviour
- H = 0.5  – random walk (Brownian motion)
- H < 0.5  – anti-persistent / mean-reverting behaviour

This module uses the Rescaled Range (R/S) analysis method over a rolling
window to produce a per-bar estimate of H.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 23.
Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
*Transactions of the American Society of Civil Engineers*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array, log_returns
from kaufman_indicators.utils.output import pandas_aware


def _rs_hurst(log_ret: np.ndarray) -> float:
    """Estimate Hurst exponent via R/S analysis on *log_ret*.

    Returns ``NaN`` if estimation is not possible.
    """
    n = len(log_ret)
    if n < 8:
        return np.nan

    lags = []
    rs_vals = []

    # Use sub-window sizes that are powers of 2 (or reasonable steps)
    lag = 4
    while lag <= n // 2:
        chunks = [log_ret[i: i + lag] for i in range(0, n - lag + 1, lag)]
        if len(chunks) < 2:
            break
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            deviation = np.cumsum(chunk - mean_c)
            r = np.max(deviation) - np.min(deviation)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(rs_chunk)))
        lag *= 2

    if len(lags) < 2:
        return np.nan

    # Hurst = slope of log(R/S) vs log(lag)
    coeffs = np.polyfit(lags, rs_vals, 1)
    return float(coeffs[0])


@pandas_aware
def hurst_exponent(prices: np.ndarray, period: int = 100) -> np.ndarray:
    """Rolling Hurst Exponent via R/S analysis.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Rolling window for the estimate (default 100; a larger window gives
        a more reliable estimate at the cost of fewer valid values).

    Returns
    -------
    np.ndarray
        Hurst exponent values; same length as *prices*; ``NaN`` for the first
        *period* entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    lr = log_returns(prices)  # length n-1

    for i in range(period, n):
        window_lr = lr[i - period: i]
        result[i] = _rs_hurst(window_lr)

    return result
