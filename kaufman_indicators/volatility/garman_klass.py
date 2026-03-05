"""Garman-Klass Volatility estimator.

Combines open, high, low, and close prices for a more efficient volatility
estimate than close-to-close or Parkinson estimators alone.

    σ² = 0.5 · [ln(H/L)]² − (2·ln2 − 1) · [ln(C/O)]²

For a rolling window of length *period*:

    σ_t = sqrt( mean( 0.5·[ln(H/L)]² − (2·ln2 − 1)·[ln(C/O)]² ) )

Reference
---------
Garman, M. B. & Klass, M. J. (1980). On the Estimation of Security Price
Volatilities from Historical Data. *Journal of Business*, 53(1), 67–78.

Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 21.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.utils.rolling import rolling_mean


@pandas_aware
def garman_klass_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Garman-Klass (OHLC) volatility estimator.

    Parameters
    ----------
    open_:
        1-D array-like of open prices.
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    close:
        1-D array-like of close prices.
    period:
        Rolling window length (default 20).
    annualize:
        If ``True`` (default), multiply by ``sqrt(periods_per_year)``.
    periods_per_year:
        Trading periods in a year (default 252).

    Returns
    -------
    np.ndarray
        Garman-Klass volatility values; ``NaN`` for the first ``period - 1``
        entries.
    """
    open_ = to_float_array(open_)
    high = to_float_array(high)
    low = to_float_array(low)
    close = to_float_array(close)

    if not (len(open_) == len(high) == len(low) == len(close)):
        raise ValueError("open, high, low, and close must have the same length")

    n = len(high)
    result = np.full(n, np.nan)

    if n < period:
        return result

    with np.errstate(invalid="ignore", divide="ignore"):
        log_hl = np.where(low > 0, np.log(high / low), np.nan)
        log_co = np.where(open_ > 0, np.log(close / open_), np.nan)

    gk_var = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

    mean_var = rolling_mean(gk_var, period)

    # Clamp to zero before sqrt to handle small negative values from
    # floating-point arithmetic.
    vol = np.sqrt(np.maximum(mean_var, 0.0))

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol
