"""Price Series Entropy.

Provides two entropy measures for price series:

1. **Approximate Entropy (ApEn)** – quantifies the unpredictability of
   fluctuations.  Higher values indicate greater randomness / less structure.

2. **Shannon Entropy** – applied to discretised return bins to measure
   information content.

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.), Chapter 23.
Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
*Proceedings of the National Academy of Sciences*.
"""

import numpy as np

from kaufman_indicators.utils.math_helpers import to_float_array, log_returns


def _approx_entropy(series: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """Compute Approximate Entropy (ApEn) for a 1-D *series*.

    Parameters
    ----------
    series:
        Normalised 1-D array.
    m:
        Template length (default 2).
    r_scale:
        Tolerance as a fraction of the series std (default 0.2).

    Returns ``NaN`` if *series* is too short.
    """
    n = len(series)
    if n < m + 1:
        return np.nan

    r = r_scale * np.std(series, ddof=1)
    if r == 0:
        return np.nan

    def _phi(m_val: int) -> float:
        templates = np.array([series[i: i + m_val] for i in range(n - m_val + 1)])
        count = np.array([
            np.sum(np.max(np.abs(templates - templates[i]), axis=1) <= r)
            for i in range(len(templates))
        ])
        count = np.maximum(count, 1)
        return np.sum(np.log(count / (n - m_val + 1))) / (n - m_val + 1)

    return _phi(m) - _phi(m + 1)


def price_entropy(
    prices: np.ndarray,
    period: int = 50,
    method: str = "shannon",
    bins: int = 10,
    apen_m: int = 2,
    apen_r: float = 0.2,
) -> np.ndarray:
    """Rolling entropy of a price series.

    Parameters
    ----------
    prices:
        1-D array-like of closing prices.
    period:
        Rolling window (default 50).
    method:
        ``"shannon"`` (default) or ``"approximate"``.
    bins:
        Number of bins for Shannon entropy discretisation (default 10).
    apen_m:
        Template length for Approximate Entropy (default 2).
    apen_r:
        Tolerance scale for Approximate Entropy (default 0.2).

    Returns
    -------
    np.ndarray
        Entropy values; same length as *prices*; ``NaN`` for the first
        *period* entries.
    """
    prices = to_float_array(prices)
    n = len(prices)
    result = np.full(n, np.nan)

    if method not in ("shannon", "approximate"):
        raise ValueError("method must be 'shannon' or 'approximate'")

    if n <= period:
        return result

    lr = log_returns(prices)  # length n-1

    for i in range(period, n):
        window = lr[i - period: i]

        if method == "shannon":
            counts, _ = np.histogram(window, bins=bins)
            counts = counts[counts > 0]
            probs = counts / counts.sum()
            result[i] = -np.sum(probs * np.log2(probs))
        else:
            result[i] = _approx_entropy(window, m=apen_m, r_scale=apen_r)

    return result
