"""QuantConnect / LEAN adapter for kaufman_indicators.

This module provides a thin bridge between the ``kaufman_indicators`` library
and the QuantConnect LEAN algorithmic trading engine.  In LEAN each indicator
is typically an instance that can be updated bar-by-bar; this adapter wraps
the vectorised numpy functions so they can be called with a rolling history
``pandas.DataFrame`` that LEAN strategies commonly produce.

Usage example (inside a LEAN ``QCAlgorithm``)::

    from adapters.lean_adapter import LeanIndicatorAdapter

    class MyAlgorithm(QCAlgorithm):
        def Initialize(self):
            self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
            self.adapter = LeanIndicatorAdapter()

        def OnData(self, data):
            history = self.History(self.symbol, 100, Resolution.Daily)
            if history.empty:
                return
            kama_val = self.adapter.kama(history["close"].values)[-1]
            rsi_val  = self.adapter.rsi(history["close"].values)[-1]
            self.Debug(f"KAMA={kama_val:.2f}  RSI={rsi_val:.2f}")

Notes
-----
- This file does **not** import any LEAN/QuantConnect modules so it can be
  used and tested independently of the LEAN runtime.
- All methods accept plain Python sequences or numpy arrays and return numpy
  arrays.
"""

from __future__ import annotations

import numpy as np
from typing import Any

import kaufman_indicators as ki


class LeanIndicatorAdapter:
    """Adapter that exposes kaufman_indicators functions with a consistent
    interface suited to QuantConnect LEAN history DataFrames.

    Each method accepts the relevant price series as plain arrays (extracted
    from a history DataFrame) and returns a numpy array of the same length.
    The last valid element of the returned array is the *current* indicator
    value.
    """

    # ------------------------------------------------------------------ #
    #  Trend                                                               #
    # ------------------------------------------------------------------ #

    def efficiency_ratio(self, close: Any, period: int = 10) -> np.ndarray:
        """Kaufman Efficiency Ratio."""
        return ki.efficiency_ratio(close, period)

    def kama(
        self,
        close: Any,
        period: int = 10,
        fast: int = 2,
        slow: int = 30,
    ) -> np.ndarray:
        """Kaufman Adaptive Moving Average."""
        return ki.kama(close, period, fast, slow)

    def sma(self, close: Any, period: int = 20) -> np.ndarray:
        """Simple Moving Average."""
        return ki.sma(close, period)

    def ema(self, close: Any, period: int = 20) -> np.ndarray:
        """Exponential Moving Average."""
        return ki.ema(close, period)

    def wma(self, close: Any, period: int = 20) -> np.ndarray:
        """Weighted Moving Average."""
        return ki.wma(close, period)

    # ------------------------------------------------------------------ #
    #  Momentum                                                            #
    # ------------------------------------------------------------------ #

    def roc(self, close: Any, period: int = 12) -> np.ndarray:
        """Rate of Change (%)."""
        return ki.roc(close, period)

    def rsi(self, close: Any, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        return ki.rsi(close, period)

    def macd(
        self,
        close: Any,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> dict[str, np.ndarray]:
        """MACD – returns a dict with keys ``macd_line``, ``signal``,
        ``histogram``."""
        result = ki.macd(close, fast, slow, signal_period)
        return {
            "macd_line": result.macd_line,
            "signal": result.signal,
            "histogram": result.histogram,
        }

    def stochastic(
        self,
        high: Any,
        low: Any,
        close: Any,
        k_period: int = 14,
        d_period: int = 3,
    ) -> dict[str, np.ndarray]:
        """Stochastic Oscillator – returns a dict with keys ``k`` and ``d``."""
        result = ki.stochastic(high, low, close, k_period, d_period)
        return {"k": result.k, "d": result.d}

    # ------------------------------------------------------------------ #
    #  Volatility                                                          #
    # ------------------------------------------------------------------ #

    def true_range(
        self, high: Any, low: Any, close: Any
    ) -> np.ndarray:
        """True Range."""
        return ki.true_range(high, low, close)

    def atr(
        self,
        high: Any,
        low: Any,
        close: Any,
        period: int = 14,
    ) -> np.ndarray:
        """Average True Range."""
        return ki.atr(high, low, close, period)

    def realized_vol(
        self,
        close: Any,
        period: int = 20,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Realized (historical) volatility."""
        return ki.realized_vol(close, period, annualize, periods_per_year)

    # ------------------------------------------------------------------ #
    #  Range                                                               #
    # ------------------------------------------------------------------ #

    def bollinger_bands(
        self,
        close: Any,
        period: int = 20,
        num_std: float = 2.0,
    ) -> dict[str, np.ndarray]:
        """Bollinger Bands – returns a dict with keys ``middle``, ``upper``,
        ``lower``, ``bandwidth``, ``percent_b``."""
        result = ki.bollinger_bands(close, period, num_std)
        return {
            "middle": result.middle,
            "upper": result.upper,
            "lower": result.lower,
            "bandwidth": result.bandwidth,
            "percent_b": result.percent_b,
        }

    def donchian_channels(
        self, high: Any, low: Any, period: int = 20
    ) -> dict[str, np.ndarray]:
        """Donchian Channels – returns a dict with keys ``upper``, ``lower``,
        ``mid``."""
        result = ki.donchian_channels(high, low, period)
        return {"upper": result.upper, "lower": result.lower, "mid": result.mid}

    def williams_r(
        self, high: Any, low: Any, close: Any, period: int = 14
    ) -> np.ndarray:
        """Williams %R."""
        return ki.williams_r(high, low, close, period)

    # ------------------------------------------------------------------ #
    #  Market Quality                                                      #
    # ------------------------------------------------------------------ #

    def fdi(self, close: Any, period: int = 30) -> np.ndarray:
        """Fractal Dimension Index."""
        return ki.fdi(close, period)

    def hurst_exponent(self, close: Any, period: int = 100) -> np.ndarray:
        """Hurst Exponent (R/S analysis)."""
        return ki.hurst_exponent(close, period)

    def price_entropy(
        self,
        close: Any,
        period: int = 50,
        method: str = "shannon",
    ) -> np.ndarray:
        """Price series entropy."""
        return ki.price_entropy(close, period, method)
