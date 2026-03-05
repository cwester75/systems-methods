"""Volume Rate of Change.

    Volume ROC = (volume_t − volume_{t−n}) / volume_{t−n} × 100

Measures the percentage change in volume over *period* bars.  Spikes in
Volume ROC often precede or confirm significant price moves.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 11.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array


def volume_roc(volume: np.ndarray, period: int = 12) -> np.ndarray:
    """Volume Rate of Change.

    Parameters
    ----------
    volume:
        1-D array-like of volume values.
    period:
        Look-back period (default 12).

    Returns
    -------
    np.ndarray
        Volume ROC values as percentages; ``NaN`` for the first *period*
        entries.
    """
    volume = to_float_array(volume)
    n = len(volume)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    prev = volume[:-period]
    with np.errstate(invalid="ignore", divide="ignore"):
        result[period:] = np.where(
            prev != 0,
            (volume[period:] - prev) / prev * 100.0,
            np.nan,
        )
    return result
