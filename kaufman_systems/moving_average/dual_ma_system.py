"""
Dual Moving Average System

Concept
-------
Classic moving-average crossover trend system.

Signal logic
------------
Fast MA crosses above Slow MA → LONG
Fast MA crosses below Slow MA → SHORT

Risk model
----------
ATR-based position sizing to normalize risk across instruments.

This system is intentionally simple because it serves as a baseline
trend-following system in the Kaufman research library.
"""

import numpy as np


class DualMASystem:

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def sma(self, prices, period):
        prices = np.asarray(prices)

        if len(prices) < period:
            return None

        return np.mean(prices[-period:])

    def atr(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(closes) < self.atr_period + 1:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        return np.mean(tr[-self.atr_period:])

    # ---------------------------------------------------------
    # Core system interface
    # ---------------------------------------------------------

    def signal(self, closes):
        """
        Returns
        -------
        1  → Long
       -1  → Short
        0  → Flat
        """

        fast_ma = self.sma(closes, self.fast_period)
        slow_ma = self.sma(closes, self.slow_period)

        if fast_ma is None or slow_ma is None:
            return 0

        if fast_ma > slow_ma:
            return 1

        if fast_ma < slow_ma:
            return -1

        return 0

    def position_sizing(self, equity, highs, lows, closes):
        """
        ATR-normalized position sizing.
        """

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * self.risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, highs, lows, closes):
        """
        Basic volatility sanity check.
        """

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, closes):
        return {
            "fast_ma": self.sma(closes, self.fast_period),
            "slow_ma": self.sma(closes, self.slow_period),
        }
