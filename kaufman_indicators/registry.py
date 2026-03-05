"""Indicator Registry – central lookup so strategies can call indicators generically.

Usage
-----
>>> from kaufman_indicators.registry import INDICATORS
>>> rsi_values = INDICATORS["rsi"](prices)
>>> atr_values = INDICATORS["atr"](high, low, close)

The ``get`` helper provides a friendlier interface with a clear error message::

>>> from kaufman_indicators.registry import get
>>> fn = get("rsi")
>>> fn(prices)
"""

from __future__ import annotations

from kaufman_indicators.trend.efficiency_ratio import efficiency_ratio
from kaufman_indicators.trend.kama import kama
from kaufman_indicators.trend.linreg import linreg, linreg_forecast
from kaufman_indicators.trend.moving_averages import sma, ema, wma, dema, tema

from kaufman_indicators.momentum.roc import roc
from kaufman_indicators.momentum.rsi import rsi
from kaufman_indicators.momentum.macd import macd
from kaufman_indicators.momentum.stochastic import stochastic
from kaufman_indicators.momentum.momentum import momentum

from kaufman_indicators.volatility.true_range import true_range
from kaufman_indicators.volatility.atr import atr
from kaufman_indicators.volatility.realized_vol import realized_vol
from kaufman_indicators.volatility.parkinson import parkinson_vol
from kaufman_indicators.volatility.garman_klass import garman_klass_vol

from kaufman_indicators.range.bollinger import bollinger_bands
from kaufman_indicators.range.donchian import donchian_channels
from kaufman_indicators.range.williams_r import williams_r
from kaufman_indicators.range.zscore import price_zscore

from kaufman_indicators.market_quality.fdi import fdi
from kaufman_indicators.market_quality.hurst import hurst_exponent
from kaufman_indicators.market_quality.entropy import price_entropy
from kaufman_indicators.market_quality.volume_roc import volume_roc
from kaufman_indicators.market_quality.volume_zscore import volume_zscore

INDICATORS: dict[str, callable] = {
    # Trend / Direction
    "efficiency_ratio": efficiency_ratio,
    "kama": kama,
    "linreg": linreg,
    "linreg_forecast": linreg_forecast,
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "dema": dema,
    "tema": tema,
    # Momentum
    "roc": roc,
    "rsi": rsi,
    "macd": macd,
    "stochastic": stochastic,
    "momentum": momentum,
    # Volatility
    "true_range": true_range,
    "atr": atr,
    "realized_vol": realized_vol,
    "parkinson_vol": parkinson_vol,
    "garman_klass_vol": garman_klass_vol,
    # Range / Position
    "bollinger_bands": bollinger_bands,
    "donchian_channels": donchian_channels,
    "williams_r": williams_r,
    "price_zscore": price_zscore,
    # Market Quality
    "fdi": fdi,
    "hurst_exponent": hurst_exponent,
    "price_entropy": price_entropy,
    "volume_roc": volume_roc,
    "volume_zscore": volume_zscore,
}


def get(name: str) -> callable:
    """Look up an indicator by name.

    Parameters
    ----------
    name:
        Key in :data:`INDICATORS` (case-sensitive).

    Returns
    -------
    callable
        The indicator function.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    try:
        return INDICATORS[name]
    except KeyError:
        available = ", ".join(sorted(INDICATORS))
        raise KeyError(
            f"Unknown indicator {name!r}. Available: {available}"
        ) from None
