"""Parkinson Volatility estimator.

Uses the high-low range to estimate volatility, which is more efficient than
close-to-close estimators when intraday extremes are available.

    σ² = (1 / (4 · n · ln2)) · Σ [ln(H_i / L_i)]²

For a rolling window of length *period*:

    σ_t = sqrt( (1 / (4 · ln2)) · mean( [ln(H / L)]² ) )

Reference
---------
Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of
the Rate of Return. *Journal of Business*, 53(1), 61–65.

Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 21.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.utils.rolling import rolling_mean


@pandas_aware
def parkinson_vol(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Parkinson (high-low) volatility estimator.

    Parameters
    ----------
    high:
        1-D array-like of high prices.
    low:
        1-D array-like of low prices.
    period:
        Rolling window length (default 20).
    annualize:
        If ``True`` (default), multiply by ``sqrt(periods_per_year)``.
    periods_per_year:
        Trading periods in a year (default 252).

    Returns
    -------
    np.ndarray
        Parkinson volatility values; ``NaN`` for the first ``period - 1``
        entries.
    """
    high = to_float_array(high)
    low = to_float_array(low)

    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    n = len(high)
    result = np.full(n, np.nan)

    if n < period:
        return result

    with np.errstate(invalid="ignore", divide="ignore"):
        log_hl_sq = np.where(
            low > 0, np.log(high / low) ** 2, np.nan
        )

    factor = 1.0 / (4.0 * np.log(2.0))
    mean_sq = rolling_mean(log_hl_sq, period)
    vol = np.sqrt(factor * mean_sq)

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol
