"""
Intraday Range Pattern System

Concept
-------
When the current bar's range exceeds a multiple of the average range
over the lookback period, it signals an expansion move.  The close
position within the bar determines direction.

Signal logic
------------
avg_range = mean(highs[-period:] - lows[-period:]) (prior bars)
current_range = highs[-1] - lows[-1]

If current_range > expansion_mult * avg_range:
    close in upper half -> LONG
    close in lower half -> SHORT
Otherwise              -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class IntradayRangePatternSystem(TradingSystem):

    def __init__(
        self,
        range_period: int = 20,
        expansion_mult: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.range_period = range_period
        self.expansion_mult = expansion_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def atr(self, highs, lows, closes):
        highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
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

    def signal(self, data):
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])
        closes = np.asarray(data["closes"])

        if len(closes) < self.range_period + 1:
            return 0

        prior_ranges = highs[-(self.range_period + 1):-1] - lows[-(self.range_period + 1):-1]
        avg_range = np.mean(prior_ranges)
        current_range = highs[-1] - lows[-1]

        if avg_range == 0:
            return 0

        if current_range > self.expansion_mult * avg_range:
            mid = (highs[-1] + lows[-1]) / 2
            if closes[-1] > mid:
                return 1
            if closes[-1] < mid:
                return -1

        return 0

    def position_sizing(self, data, risk):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return 0
        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        if len(highs) < self.range_period + 1:
            return {"avg_range": None, "current_range": None, "expansion_ratio": None}

        prior_ranges = highs[-(self.range_period + 1):-1] - lows[-(self.range_period + 1):-1]
        avg_range = float(np.mean(prior_ranges))
        current_range = float(highs[-1] - lows[-1])
        ratio = current_range / avg_range if avg_range != 0 else None

        return {"avg_range": avg_range, "current_range": current_range, "expansion_ratio": ratio}
