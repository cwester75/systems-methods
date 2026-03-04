"""Rolling-window utility functions."""

import numpy as np


def rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    """Return a 2-D array of rolling windows over *a*.

    Parameters
    ----------
    a:
        1-D array of values.
    window:
        Number of periods in each window.

    Returns
    -------
    np.ndarray
        Shape ``(len(a) - window + 1, window)``.  Row ``i`` contains
        ``a[i : i + window]``.

    Raises
    ------
    ValueError
        If *window* is larger than ``len(a)``.
    """
    a = np.asarray(a, dtype=float)
    if window > len(a):
        raise ValueError(
            f"window ({window}) must be <= length of input ({len(a)})"
        )
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling arithmetic mean.

    Returns an array the same length as *a*, with ``NaN`` for positions
    that do not yet have a full window.
    """
    a = np.asarray(a, dtype=float)
    result = np.full(len(a), np.nan)
    if window > len(a):
        return result
    windows = rolling_window(a, window)
    result[window - 1:] = windows.mean(axis=1)
    return result


def rolling_std(a: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Rolling standard deviation.

    Parameters
    ----------
    a:
        1-D array of values.
    window:
        Number of periods.
    ddof:
        Delta degrees of freedom (default 1 → sample std).

    Returns an array the same length as *a*, with ``NaN`` for incomplete
    windows.
    """
    a = np.asarray(a, dtype=float)
    result = np.full(len(a), np.nan)
    if window > len(a):
        return result
    windows = rolling_window(a, window)
    result[window - 1:] = windows.std(axis=1, ddof=ddof)
    return result


def rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum."""
    a = np.asarray(a, dtype=float)
    result = np.full(len(a), np.nan)
    if window > len(a):
        return result
    windows = rolling_window(a, window)
    result[window - 1:] = windows.max(axis=1)
    return result


def rolling_min(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum."""
    a = np.asarray(a, dtype=float)
    result = np.full(len(a), np.nan)
    if window > len(a):
        return result
    windows = rolling_window(a, window)
    result[window - 1:] = windows.min(axis=1)
    return result


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum."""
    a = np.asarray(a, dtype=float)
    result = np.full(len(a), np.nan)
    if window > len(a):
        return result
    windows = rolling_window(a, window)
    result[window - 1:] = windows.sum(axis=1)
    return result
