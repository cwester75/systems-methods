"""
Seasonal Pattern System

Concept
-------
Detects seasonal (periodic) return patterns using a fixed cycle
period.  Compares the current price to its value one cycle ago
to identify seasonal uptrends and downtrends.

Signal logic
------------
If close[-1] > close[-cycle_period]  -> LONG  (seasonal uptrend)
If close[-1] < close[-cycle_period]  -> SHORT (seasonal downtrend)
Otherwise                            -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class SeasonalPatternSystem(TradingSystem):

    def __init__(
        self,
        cycle_period: int = 252,
        min_bars: int = 50,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.cycle_period = cycle_period
        self.min_bars = min_bars
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
        closes = np.asarray(data["closes"])

        required = max(self.cycle_period + 1, self.min_bars)
        if len(closes) < required:
            return 0

        past_price = closes[-(self.cycle_period + 1)]
        if past_price == 0:
            return 0

        if closes[-1] > past_price:
            return 1
        if closes[-1] < past_price:
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
        closes = np.asarray(data["closes"])
        required = max(self.cycle_period + 1, self.min_bars)
        if len(closes) < required:
            return {"seasonal_return": None, "cycle_period": self.cycle_period}
        past_price = closes[-(self.cycle_period + 1)]
        seasonal_ret = float((closes[-1] - past_price) / past_price) if past_price != 0 else None
        return {"seasonal_return": seasonal_ret, "cycle_period": self.cycle_period}
