"""Volume Z-Score.

    Z = (volume − SMA(volume, period)) / rolling_std(volume, period)

Measures how many standard deviations the current volume is from its rolling
mean.  High positive values signal unusually heavy volume; negative values
signal unusually light volume.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 20.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array
from kaufman_indicators.utils.output import pandas_aware
from kaufman_indicators.utils.rolling import rolling_mean, rolling_std


@pandas_aware
def volume_zscore(
    volume: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Rolling Volume Z-Score.

    Parameters
    ----------
    volume:
        1-D array-like of volume values.
    period:
        Rolling window for the mean and standard deviation (default 20).

    Returns
    -------
    np.ndarray
        Z-Score values; ``NaN`` for the first ``period - 1`` entries.
    """
    volume = to_float_array(volume)
    n = len(volume)
    result = np.full(n, np.nan)

    if n < period:
        return result

    mean = rolling_mean(volume, period)
    std = rolling_std(volume, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(std != 0, (volume - mean) / std, 0.0)

    # Restore NaN for warmup period
    result[:period - 1] = np.nan

    return result
