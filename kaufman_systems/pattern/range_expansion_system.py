"""
Range Expansion System

Concept
-------
Signals when the current bar's range (high - low) expands significantly
relative to the recent average range.  A large expansion often marks the
start of a new directional move.

Signal logic
------------
If current range > expansion_mult × average range (N bars):
  Close > Open (or close > prev close) → LONG
  Close < Open (or close < prev close) → SHORT
Otherwise                              → FLAT

Since we only have close/high/low (no open), we use close direction
relative to the previous close to determine signal direction.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RangeExpansionSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 10,
        expansion_mult: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.expansion_mult = expansion_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def range_expansion(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.lookback + 1:
            return None, None

        ranges = highs - lows
        avg_range = np.mean(ranges[-(self.lookback + 1):-1])
        current_range = ranges[-1]

        return current_range, avg_range

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
        closes = np.asarray(closes)

        if len(closes) < 2:
            return 0

        current_range, avg_range = self.range_expansion(highs, lows)

        if current_range is None:
            return 0

        if avg_range == 0:
            return 0

        if current_range <= self.expansion_mult * avg_range:
            return 0

        if closes[-1] > closes[-2]:
            return 1

        if closes[-1] < closes[-2]:
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

        current_range, avg_range = self.range_expansion(highs, lows)

        return {
            "current_range": current_range,
            "avg_range": avg_range,
            "expansion_ratio": current_range / avg_range if avg_range and avg_range != 0 else None,
        }
