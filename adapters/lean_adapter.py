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

    def dema(self, close: Any, period: int = 20) -> np.ndarray:
        """Double Exponential Moving Average."""
        return ki.dema(close, period)

    def tema(self, close: Any, period: int = 20) -> np.ndarray:
        """Triple Exponential Moving Average."""
        return ki.tema(close, period)

    def linreg(self, close: Any, period: int = 14) -> dict[str, np.ndarray]:
        """Linear Regression – returns a dict with keys ``value``, ``slope``,
        ``intercept``, ``r_squared``."""
        result = ki.linreg(close, period)
        return {
            "value": result.value,
            "slope": result.slope,
            "intercept": result.intercept,
            "r_squared": result.r_squared,
        }

    def linreg_forecast(
        self, close: Any, period: int = 14, offset: int = 1
    ) -> np.ndarray:
        """Linear Regression Forecast."""
        return ki.linreg_forecast(close, period, offset)

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

    def momentum(self, close: Any, period: int = 10) -> np.ndarray:
        """Momentum (absolute price change)."""
        return ki.momentum(close, period)

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

    def parkinson_vol(
        self,
        high: Any,
        low: Any,
        period: int = 20,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Parkinson (high-low) volatility."""
        return ki.parkinson_vol(high, low, period, annualize, periods_per_year)

    def garman_klass_vol(
        self,
        open_: Any,
        high: Any,
        low: Any,
        close: Any,
        period: int = 20,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> np.ndarray:
        """Garman-Klass (OHLC) volatility."""
        return ki.garman_klass_vol(
            open_, high, low, close, period, annualize, periods_per_year
        )

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

    def price_zscore(self, close: Any, period: int = 20) -> np.ndarray:
        """Price Z-Score."""
        return ki.price_zscore(close, period)

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

    def volume_roc(self, volume: Any, period: int = 12) -> np.ndarray:
        """Volume Rate of Change (%)."""
        return ki.volume_roc(volume, period)

    def volume_zscore(self, volume: Any, period: int = 20) -> np.ndarray:
        """Volume Z-Score."""
        return ki.volume_zscore(volume, period)


class IndicatorLibrary:
    """Batch indicator computation for QCAlgorithm data streams.

    Provides a single :meth:`compute` call that accepts a LEAN history
    DataFrame and returns a dictionary of all computed indicators.

    Usage example (inside a LEAN ``QCAlgorithm``)::

        from adapters.lean_adapter import IndicatorLibrary

        class MyAlgorithm(QCAlgorithm):
            def Initialize(self):
                self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
                self.lib = IndicatorLibrary(self)

            def OnData(self, data):
                history = self.History(self.symbol, 200, Resolution.Daily)
                if history.empty:
                    return
                indicators = self.lib.compute(history)
                self.Debug(f"RSI={indicators['rsi'][-1]:.2f}")

    Parameters
    ----------
    algorithm:
        The ``QCAlgorithm`` instance.  Stored for potential future use
        (e.g. logging via ``algorithm.Debug``), but not required for
        computation.
    """

    def __init__(self, algorithm: Any = None) -> None:
        self.algorithm = algorithm

    def compute(self, history: Any) -> dict[str, Any]:
        """Compute all indicators from a LEAN history DataFrame.

        Parameters
        ----------
        history:
            A ``pandas.DataFrame`` with columns ``close``, ``high``,
            ``low``, and optionally ``open`` and ``volume``.

        Returns
        -------
        dict[str, Any]
            Dictionary mapping indicator names to numpy arrays (or dicts
            for multi-output indicators like MACD, Bollinger, etc.).
        """
        close = np.asarray(history["close"], dtype=float)
        high = np.asarray(history["high"], dtype=float)
        low = np.asarray(history["low"], dtype=float)

        has_open = "open" in history.columns
        has_volume = "volume" in history.columns

        indicators: dict[str, Any] = {}

        # -- Trend --------------------------------------------------------
        indicators["efficiency_ratio"] = ki.efficiency_ratio(close)
        indicators["kama"] = ki.kama(close)
        indicators["sma"] = ki.sma(close, period=20)
        indicators["ema"] = ki.ema(close, period=20)

        lr = ki.linreg(close)
        indicators["linreg_slope"] = lr.slope
        indicators["linreg_intercept"] = lr.intercept
        indicators["linreg_r_squared"] = lr.r_squared

        # -- Momentum -----------------------------------------------------
        indicators["roc"] = ki.roc(close)
        indicators["rsi"] = ki.rsi(close)
        indicators["momentum"] = ki.momentum(close)

        macd_result = ki.macd(close)
        indicators["macd"] = macd_result.macd_line
        indicators["macd_signal"] = macd_result.signal
        indicators["macd_histogram"] = macd_result.histogram

        indicators["stochastic_k"] = ki.stochastic(high, low, close).k
        indicators["stochastic_d"] = ki.stochastic(high, low, close).d

        # -- Volatility ---------------------------------------------------
        indicators["atr"] = ki.atr(high, low, close)
        indicators["true_range"] = ki.true_range(high, low, close)
        indicators["realized_vol"] = ki.realized_vol(close)
        indicators["parkinson_vol"] = ki.parkinson_vol(high, low)

        if has_open:
            open_ = np.asarray(history["open"], dtype=float)
            indicators["garman_klass_vol"] = ki.garman_klass_vol(
                open_, high, low, close
            )

        # -- Range --------------------------------------------------------
        bb = ki.bollinger_bands(close)
        indicators["bollinger_upper"] = bb.upper
        indicators["bollinger_middle"] = bb.middle
        indicators["bollinger_lower"] = bb.lower
        indicators["bollinger_bandwidth"] = bb.bandwidth
        indicators["bollinger_percent_b"] = bb.percent_b

        dc = ki.donchian_channels(high, low)
        indicators["donchian_upper"] = dc.upper
        indicators["donchian_lower"] = dc.lower
        indicators["donchian_mid"] = dc.mid

        indicators["williams_r"] = ki.williams_r(high, low, close)
        indicators["price_zscore"] = ki.price_zscore(close)

        # -- Market Quality -----------------------------------------------
        indicators["fdi"] = ki.fdi(close)
        indicators["hurst_exponent"] = ki.hurst_exponent(close)
        indicators["entropy"] = ki.price_entropy(close)

        if has_volume:
            volume = np.asarray(history["volume"], dtype=float)
            indicators["volume_roc"] = ki.volume_roc(volume)
            indicators["volume_zscore"] = ki.volume_zscore(volume)

        return indicators


class LeanSystemAdapter:
    """Bridge between any :class:`TradingSystem` and a LEAN ``QCAlgorithm``.

    Translates LEAN history DataFrames into the standardised ``data`` / ``risk``
    dicts that every trading system expects, so integration requires no
    system-specific glue code.

    Usage example (inside a LEAN ``QCAlgorithm``)::

        from adapters.lean_adapter import LeanSystemAdapter
        from kaufman_systems.trend.er_trend_system import ERTrendSystem

        class MyAlgorithm(QCAlgorithm):
            def Initialize(self):
                self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
                self.system = LeanSystemAdapter(ERTrendSystem())

            def OnData(self, data):
                history = self.History(self.symbol, 100, Resolution.Daily)
                if history.empty:
                    return
                sig = self.system.signal(history)
                if sig == 1:
                    self.SetHoldings(self.symbol, 1.0)
                elif sig == -1:
                    self.Liquidate(self.symbol)

    Parameters
    ----------
    system:
        Any object conforming to the :class:`TradingSystem` interface.
    equity:
        Default account equity for position sizing.
    risk_per_trade:
        Default fraction of equity risked per trade.
    """

    def __init__(
        self,
        system: Any,
        equity: float = 100_000.0,
        risk_per_trade: float = 0.01,
    ) -> None:
        self.system = system
        self.equity = equity
        self.risk_per_trade = risk_per_trade

    @staticmethod
    def _to_data(history: Any) -> dict[str, np.ndarray]:
        """Convert a LEAN history DataFrame to the standard data dict."""
        return {
            "closes": np.asarray(history["close"], dtype=float),
            "highs": np.asarray(history["high"], dtype=float),
            "lows": np.asarray(history["low"], dtype=float),
        }

    def _risk(self, equity: float | None = None) -> dict[str, float]:
        return {
            "equity": equity if equity is not None else self.equity,
            "risk_per_trade": self.risk_per_trade,
        }

    def signal(self, history: Any) -> int:
        """Generate a trading signal from a LEAN history DataFrame."""
        return self.system.signal(self._to_data(history))

    def position_sizing(
        self, history: Any, equity: float | None = None
    ) -> float:
        """Calculate position size from a LEAN history DataFrame."""
        return self.system.position_sizing(
            self._to_data(history), self._risk(equity)
        )

    def risk_filter(self, history: Any) -> bool:
        """Run the risk filter on a LEAN history DataFrame."""
        return self.system.risk_filter(self._to_data(history))

    def indicators(self, history: Any) -> dict:
        """Return diagnostic indicators from a LEAN history DataFrame."""
        return self.system.indicators(self._to_data(history))
