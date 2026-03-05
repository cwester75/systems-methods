"""Indicator Metadata Registry.

Provides structured metadata for every indicator: category, required inputs,
and default parameters.  Useful for building generic UIs, validation layers,
and strategy configuration without hard-coding indicator details.

Usage
-----
>>> from kaufman_indicators.registry_meta import INDICATOR_META
>>> meta = INDICATOR_META["rsi"]
>>> meta["category"]
'momentum'
>>> meta["required_inputs"]
['prices']
>>> meta["default_params"]
{'period': 14}
"""

from __future__ import annotations

from typing import Any

# ── Metadata for every registered indicator ──────────────────────────────────
#
# Each entry maps an indicator name to:
#   category        – one of trend, momentum, volatility, range, market_quality
#   required_inputs – positional array-like arguments the function expects
#   default_params  – keyword arguments with their default values
#                     (empty dict when the function has no optional parameters)

INDICATOR_META: dict[str, dict[str, Any]] = {
    # ── Trend / Direction ────────────────────────────────────────────────
    "efficiency_ratio": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {"period": 10},
    },
    "kama": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {"period": 10, "fast": 2, "slow": 30},
    },
    "linreg": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {"period": 14},
    },
    "linreg_forecast": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {"period": 14, "offset": 1},
    },
    "sma": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {},
    },
    "ema": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {"alpha": None},
    },
    "wma": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {},
    },
    "dema": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {},
    },
    "tema": {
        "category": "trend",
        "required_inputs": ["prices"],
        "default_params": {},
    },
    # ── Momentum ─────────────────────────────────────────────────────────
    "roc": {
        "category": "momentum",
        "required_inputs": ["prices"],
        "default_params": {"period": 12},
    },
    "rsi": {
        "category": "momentum",
        "required_inputs": ["prices"],
        "default_params": {"period": 14},
    },
    "macd": {
        "category": "momentum",
        "required_inputs": ["prices"],
        "default_params": {"fast": 12, "slow": 26, "signal_period": 9},
    },
    "stochastic": {
        "category": "momentum",
        "required_inputs": ["high", "low", "close"],
        "default_params": {"k_period": 14, "d_period": 3},
    },
    "momentum": {
        "category": "momentum",
        "required_inputs": ["prices"],
        "default_params": {"period": 10},
    },
    # ── Volatility ───────────────────────────────────────────────────────
    "true_range": {
        "category": "volatility",
        "required_inputs": ["high", "low", "close"],
        "default_params": {},
    },
    "atr": {
        "category": "volatility",
        "required_inputs": ["high", "low", "close"],
        "default_params": {"period": 14},
    },
    "realized_vol": {
        "category": "volatility",
        "required_inputs": ["prices"],
        "default_params": {
            "period": 20,
            "annualize": True,
            "periods_per_year": 252,
        },
    },
    "parkinson_vol": {
        "category": "volatility",
        "required_inputs": ["high", "low"],
        "default_params": {
            "period": 20,
            "annualize": True,
            "periods_per_year": 252,
        },
    },
    "garman_klass_vol": {
        "category": "volatility",
        "required_inputs": ["open", "high", "low", "close"],
        "default_params": {
            "period": 20,
            "annualize": True,
            "periods_per_year": 252,
        },
    },
    # ── Range / Position ─────────────────────────────────────────────────
    "bollinger_bands": {
        "category": "range",
        "required_inputs": ["prices"],
        "default_params": {"period": 20, "num_std": 2.0},
    },
    "donchian_channels": {
        "category": "range",
        "required_inputs": ["high", "low"],
        "default_params": {"period": 20},
    },
    "williams_r": {
        "category": "range",
        "required_inputs": ["high", "low", "close"],
        "default_params": {"period": 14},
    },
    "price_zscore": {
        "category": "range",
        "required_inputs": ["prices"],
        "default_params": {"period": 20},
    },
    # ── Market Quality ───────────────────────────────────────────────────
    "fdi": {
        "category": "market_quality",
        "required_inputs": ["prices"],
        "default_params": {"period": 30},
    },
    "hurst_exponent": {
        "category": "market_quality",
        "required_inputs": ["prices"],
        "default_params": {"period": 100},
    },
    "price_entropy": {
        "category": "market_quality",
        "required_inputs": ["prices"],
        "default_params": {
            "period": 50,
            "method": "shannon",
            "bins": 10,
            "apen_m": 2,
            "apen_r": 0.2,
        },
    },
    "volume_roc": {
        "category": "market_quality",
        "required_inputs": ["volume"],
        "default_params": {"period": 12},
    },
    "volume_zscore": {
        "category": "market_quality",
        "required_inputs": ["volume"],
        "default_params": {"period": 20},
    },
}


def get_meta(name: str) -> dict[str, Any]:
    """Look up metadata for an indicator by name.

    Parameters
    ----------
    name:
        Key in :data:`INDICATOR_META` (case-sensitive).

    Returns
    -------
    dict
        Metadata dictionary with keys ``category``, ``required_inputs``,
        and ``default_params``.

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
        if required_input in meta["required_inputs"]
    ]
