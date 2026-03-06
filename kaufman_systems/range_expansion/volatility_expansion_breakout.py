"""
Volatility Expansion Breakout System

Concept
-------
Detects when the current bar's range expands significantly compared to
a short-term average range, AND the close is near the bar's extreme.
This combines range expansion with directional conviction (close near
high = bullish, close near low = bearish).

Signal logic
------------
If current range > expansion_mult × avg_range(lookback):
  If close in upper 25% of bar → LONG
  If close in lower 25% of bar → SHORT
Otherwise                      → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class VolatilityExpansionBreakoutSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 10,
        expansion_mult: float = 1.5,
        close_pct: float = 0.25,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.expansion_mult = expansion_mult
        self.close_pct = close_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

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
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(highs) < self.lookback + 1:
            return 0

        ranges = highs - lows
        avg_range = np.mean(ranges[-(self.lookback + 1):-1])
        current_range = ranges[-1]

        if avg_range == 0:
            return 0

        if current_range <= self.expansion_mult * avg_range:
            return 0

        if current_range == 0:
            return 0

        close_position = (closes[-1] - lows[-1]) / current_range

        if close_position >= (1 - self.close_pct):
            return 1

        if close_position <= self.close_pct:
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
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.lookback + 1:
            return {"current_range": None, "avg_range": None, "expansion_ratio": None}

        ranges = highs - lows
        avg_range = np.mean(ranges[-(self.lookback + 1):-1])
        current_range = ranges[-1]

        return {
            "current_range": current_range,
            "avg_range": avg_range,
            "expansion_ratio": current_range / avg_range if avg_range != 0 else None,
        }
