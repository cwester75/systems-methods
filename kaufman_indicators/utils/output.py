"""Output standardization utilities.

When the caller passes a ``pandas.Series`` as the first array-like input,
the decorator :func:`pandas_aware` automatically returns a ``pd.Series``
with:

* the original **index** preserved
* a **name** attribute set to the indicator function name

When the input is a plain numpy array (or list), the output remains a numpy
array — fully backward compatible.

Pandas is imported lazily and is **not** required at install time.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np


def _has_pandas() -> bool:
    """Return True if pandas is importable."""
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_index(args: tuple, kwargs: dict) -> Any | None:
    """Return the pandas Index from the first Series found in *args*, or None."""
    try:
        import pandas as pd
    except ImportError:
        return None

    # Check positional args first
    for arg in args:
        if isinstance(arg, pd.Series):
            return arg.index
    # Then keyword args
    for arg in kwargs.values():
        if isinstance(arg, pd.Series):
            return arg.index
    return None


def _wrap_output(result: Any, name: str, index: Any) -> Any:
    """Wrap *result* in pd.Series (or wrap NamedTuple fields) using *index*."""
    import pandas as pd

    if isinstance(result, np.ndarray):
        return pd.Series(result, index=index, name=name)

    # NamedTuple — wrap each field
    if hasattr(result, '_fields'):
        wrapped = {}
        for field in result._fields:
            arr = getattr(result, field)
            if isinstance(arr, np.ndarray):
                wrapped[field] = pd.Series(arr, index=index, name=f"{name}_{field}")
            else:
                wrapped[field] = arr
        return type(result)(**wrapped)

    return result


def pandas_aware(fn):
    """Decorator that converts numpy outputs to pd.Series when inputs are Series.

    Designed for indicator functions whose first positional argument(s) are
    array-like price/volume data.  If *any* positional or keyword array
    argument is a ``pandas.Series``, the output is wrapped with matching
    index and ``name=<function_name>``.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        index = _extract_index(args, kwargs)
        result = fn(*args, **kwargs)
        if index is not None:
            return _wrap_output(result, fn.__name__, index)
        return result

    return wrapper
