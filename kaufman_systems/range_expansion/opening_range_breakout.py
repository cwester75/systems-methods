"""
Opening Range Breakout System

Concept
-------
Uses the first N bars of a session to define the "opening range."
A breakout beyond this range signals a directional move.  Since we
work with daily bar data (no intraday), we approximate this as the
range of the first N bars from the start of the data window.

Signal logic
------------
If close > max(highs[:N]) → LONG
If close < min(lows[:N])  → SHORT
Otherwise                 → FLAT

In practice, the opening range is recalculated over the first N bars
of the lookback window, and the current close is compared to that range.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OpeningRangeBreakoutSystem(TradingSystem):

    def __init__(
        self,
        opening_bars: int = 5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.opening_bars = opening_bars
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def opening_range(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.opening_bars + 1:
            return None, None

        or_high = np.max(highs[-self.opening_bars - 1:-1])
        or_low = np.min(lows[-self.opening_bars - 1:-1])

        return or_high, or_low

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
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        or_high, or_low = self.opening_range(highs, lows)

        if or_high is None:
            return 0

        price = np.asarray(closes)[-1]

        if price > or_high:
            return 1

        if price < or_low:
            return -1

        return 0

    def position_sizing(self, data, risk):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None or atr <= 0:
            return False

        return True

    def indicators(self, data):
        highs = data["highs"]
        lows = data["lows"]

        or_high, or_low = self.opening_range(highs, lows)

        return {
            "or_high": or_high,
            "or_low": or_low,
        }
