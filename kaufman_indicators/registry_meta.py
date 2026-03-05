"""Indicator Metadata Registry.

Provides structured metadata for every indicator: category, required inputs,
parameter schemas with types, and output field descriptions.  Useful for
building generic UIs, validation layers, strategy configuration, and
runtime introspection without hard-coding indicator details.

Usage
-----
>>> from kaufman_indicators.registry_meta import INDICATOR_META, get_meta
>>> meta = get_meta("rsi")
>>> meta["category"]
'momentum'
>>> meta["inputs"]
['prices']
>>> meta["params"]
{'period': {'type': 'int', 'default': 14, 'required': False}}
>>> meta["output"]
'array'

Introspection
-------------
>>> from kaufman_indicators.registry_meta import schema
>>> schema("bollinger_bands")
{'name': 'bollinger_bands', 'category': 'range', 'inputs': ['prices'],
 'params': {...}, 'output': ['middle', 'upper', 'lower', 'bandwidth', 'percent_b']}

Validation
----------
>>> from kaufman_indicators.registry_meta import validate_meta
>>> validate_meta()  # raises if metadata drifts from actual signatures
"""

from __future__ import annotations

import inspect
from typing import Any

# ── Metadata for every registered indicator ──────────────────────────────────
#
# Each entry maps an indicator name to:
#   category – one of trend, momentum, volatility, range, market_quality
#   inputs   – positional array-like arguments the function expects
#   params   – keyword parameters with type, default, and required flag
#   output   – "array" for single-output, or list of field names for NamedTuples

INDICATOR_META: dict[str, dict[str, Any]] = {
    # ── Trend / Direction ────────────────────────────────────────────────
    "efficiency_ratio": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 10, "required": False},
        },
        "output": "array",
    },
    "kama": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 10, "required": False},
            "fast": {"type": "int", "default": 2, "required": False},
            "slow": {"type": "int", "default": 30, "required": False},
        },
        "output": "array",
    },
    "linreg": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 14, "required": False},
        },
        "output": ["value", "slope", "intercept", "r_squared"],
    },
    "linreg_forecast": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 14, "required": False},
            "offset": {"type": "int", "default": 1, "required": False},
        },
        "output": "array",
    },
    "sma": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": None, "required": True},
        },
        "output": "array",
    },
    "ema": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": None, "required": True},
            "alpha": {"type": "float", "default": None, "required": False},
        },
        "output": "array",
    },
    "wma": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": None, "required": True},
        },
        "output": "array",
    },
    "dema": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": None, "required": True},
        },
        "output": "array",
    },
    "tema": {
        "category": "trend",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": None, "required": True},
        },
        "output": "array",
    },
    # ── Momentum ─────────────────────────────────────────────────────────
    "roc": {
        "category": "momentum",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 12, "required": False},
        },
        "output": "array",
    },
    "rsi": {
        "category": "momentum",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 14, "required": False},
        },
        "output": "array",
    },
    "macd": {
        "category": "momentum",
        "inputs": ["prices"],
        "params": {
            "fast": {"type": "int", "default": 12, "required": False},
            "slow": {"type": "int", "default": 26, "required": False},
            "signal_period": {"type": "int", "default": 9, "required": False},
        },
        "output": ["macd_line", "signal", "histogram"],
    },
    "stochastic": {
        "category": "momentum",
        "inputs": ["high", "low", "close"],
        "params": {
            "k_period": {"type": "int", "default": 14, "required": False},
            "d_period": {"type": "int", "default": 3, "required": False},
        },
        "output": ["k", "d"],
    },
    "momentum": {
        "category": "momentum",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 10, "required": False},
        },
        "output": "array",
    },
    # ── Volatility ───────────────────────────────────────────────────────
    "true_range": {
        "category": "volatility",
        "inputs": ["high", "low", "close"],
        "params": {},
        "output": "array",
    },
    "atr": {
        "category": "volatility",
        "inputs": ["high", "low", "close"],
        "params": {
            "period": {"type": "int", "default": 14, "required": False},
        },
        "output": "array",
    },
    "realized_vol": {
        "category": "volatility",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
            "annualize": {"type": "bool", "default": True, "required": False},
            "periods_per_year": {"type": "int", "default": 252, "required": False},
        },
        "output": "array",
    },
    "parkinson_vol": {
        "category": "volatility",
        "inputs": ["high", "low"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
            "annualize": {"type": "bool", "default": True, "required": False},
            "periods_per_year": {"type": "int", "default": 252, "required": False},
        },
        "output": "array",
    },
    "garman_klass_vol": {
        "category": "volatility",
        "inputs": ["open", "high", "low", "close"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
            "annualize": {"type": "bool", "default": True, "required": False},
            "periods_per_year": {"type": "int", "default": 252, "required": False},
        },
        "output": "array",
    },
    # ── Range / Position ─────────────────────────────────────────────────
    "bollinger_bands": {
        "category": "range",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
            "num_std": {"type": "float", "default": 2.0, "required": False},
        },
        "output": ["middle", "upper", "lower", "bandwidth", "percent_b"],
    },
    "donchian_channels": {
        "category": "range",
        "inputs": ["high", "low"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
        },
        "output": ["upper", "lower", "mid"],
    },
    "williams_r": {
        "category": "range",
        "inputs": ["high", "low", "close"],
        "params": {
            "period": {"type": "int", "default": 14, "required": False},
        },
        "output": "array",
    },
    "price_zscore": {
        "category": "range",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
        },
        "output": "array",
    },
    # ── Market Quality ───────────────────────────────────────────────────
    "fdi": {
        "category": "market_quality",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 30, "required": False},
        },
        "output": "array",
    },
    "hurst_exponent": {
        "category": "market_quality",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 100, "required": False},
        },
        "output": "array",
    },
    "price_entropy": {
        "category": "market_quality",
        "inputs": ["prices"],
        "params": {
            "period": {"type": "int", "default": 50, "required": False},
            "method": {"type": "str", "default": "shannon", "required": False},
            "bins": {"type": "int", "default": 10, "required": False},
            "apen_m": {"type": "int", "default": 2, "required": False},
            "apen_r": {"type": "float", "default": 0.2, "required": False},
        },
        "output": "array",
    },
    "volume_roc": {
        "category": "market_quality",
        "inputs": ["volume"],
        "params": {
            "period": {"type": "int", "default": 12, "required": False},
        },
        "output": "array",
    },
    "volume_zscore": {
        "category": "market_quality",
        "inputs": ["volume"],
        "params": {
            "period": {"type": "int", "default": 20, "required": False},
        },
        "output": "array",
    },
}


# ── Lookup helpers ───────────────────────────────────────────────────────────

def get_meta(name: str) -> dict[str, Any]:
    """Look up metadata for an indicator by name.

    Parameters
    ----------
    name:
        Key in :data:`INDICATOR_META` (case-sensitive).

    Returns
    -------
    dict
        Metadata dictionary with keys ``category``, ``inputs``,
        ``params``, and ``output``.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    try:
        return INDICATOR_META[name]
    except KeyError:
        available = ", ".join(sorted(INDICATOR_META))
        raise KeyError(
            f"Unknown indicator {name!r}. Available: {available}"
        ) from None


def list_by_category(category: str) -> list[str]:
    """Return indicator names belonging to *category*."""
    return [
        name
        for name, meta in INDICATOR_META.items()
        if meta["category"] == category
    ]


def list_by_input(required_input: str) -> list[str]:
    """Return indicator names that require *required_input*."""
    return [
        name
        for name, meta in INDICATOR_META.items()
        if required_input in meta["inputs"]
    ]


# ── Schema & introspection ──────────────────────────────────────────────────

def schema(name: str) -> dict[str, Any]:
    """Return a JSON-serializable schema for indicator *name*.

    Includes the indicator name, category, inputs, full parameter
    definitions, and output description.

    >>> schema("rsi")  # doctest: +SKIP
    {'name': 'rsi', 'category': 'momentum', 'inputs': ['prices'],
     'params': {'period': {'type': 'int', 'default': 14, 'required': False}},
     'output': 'array'}
    """
    meta = get_meta(name)
    return {"name": name, **meta}


def defaults(name: str) -> dict[str, Any]:
    """Return a dict of ``{param: default}`` for optional params of *name*.

    Only includes parameters that have a non-``None`` default value.

    >>> defaults("kama")
    {'period': 10, 'fast': 2, 'slow': 30}
    """
    meta = get_meta(name)
    return {
        k: v["default"]
        for k, v in meta["params"].items()
        if not v["required"] and v["default"] is not None
    }


def required_params(name: str) -> list[str]:
    """Return parameter names that must be supplied (no default) for *name*.

    >>> required_params("sma")
    ['period']
    >>> required_params("rsi")
    []
    """
    meta = get_meta(name)
    return [k for k, v in meta["params"].items() if v["required"]]


def output_fields(name: str) -> list[str] | None:
    """Return NamedTuple field names for multi-output indicators, or None.

    >>> output_fields("bollinger_bands")
    ['middle', 'upper', 'lower', 'bandwidth', 'percent_b']
    >>> output_fields("rsi") is None
    True
    """
    meta = get_meta(name)
    out = meta["output"]
    if isinstance(out, list):
        return out
    return None


def validate_meta() -> None:
    """Validate that INDICATOR_META matches actual function signatures.

    Compares parameter names and defaults in the metadata against the
    real function signatures obtained via ``inspect``.  Raises
    ``AssertionError`` on mismatch.

    Intended for use in tests to catch metadata drift.
    """
    from kaufman_indicators.registry import INDICATORS

    for name, meta in INDICATOR_META.items():
        assert name in INDICATORS, f"{name!r} in metadata but not in INDICATORS"

        fn = INDICATORS[name]
        sig = inspect.signature(fn)
        sig_params = {
            k: v
            for k, v in sig.parameters.items()
            if k not in ("self",)
        }

        # Separate signature into positional-no-default and keyword-with-default
        sig_no_default = [
            k for k, v in sig_params.items()
            if v.default is inspect.Parameter.empty
        ]
        sig_with_default = {
            k: v.default
            for k, v in sig_params.items()
            if v.default is not inspect.Parameter.empty
        }

        # Required meta params are those without defaults in the signature
        meta_required = {k for k, v in meta["params"].items() if v["required"]}
        meta_optional = {k for k, v in meta["params"].items() if not v["required"]}

        # inputs + required params should cover all no-default sig params
        input_names = set(meta["inputs"])
        # Map input names to the actual positional param names (e.g. "open" -> "open_")
        expected_no_default = input_names | meta_required
        # Some params like "open" may be "open_" in the signature
        actual_no_default = set(sig_no_default)
        assert len(expected_no_default) == len(actual_no_default), (
            f"{name}: inputs+required {expected_no_default} vs "
            f"signature no-default {actual_no_default}"
        )

        # Optional meta params should match keyword params in signature
        assert meta_optional == set(sig_with_default.keys()), (
            f"{name}: optional params {meta_optional} vs "
            f"signature defaults {set(sig_with_default)}"
        )

        # Verify default values match
        for pname in meta_optional:
            expected = meta["params"][pname]["default"]
            actual = sig_with_default[pname]
            assert expected == actual, (
                f"{name}.{pname}: default {expected!r} vs "
                f"signature {actual!r}"
            )

    for name in INDICATORS:
        assert name in INDICATOR_META, (
            f"{name!r} in INDICATORS but not in metadata"
        )
