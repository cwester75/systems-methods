"""
Monthly Turn (Turn-of-the-Month) Pattern System

Concept
-------
Equity markets tend to produce positive returns near month
boundaries.  Using a 21-bar cycle (approximate trading month),
positions near the cycle boundary trigger a long bias.

Signal logic
------------
bar_position = len(closes) % cycle_length
If position < turn_bars OR position >= cycle_length - turn_bars:
    -> LONG
Otherwise -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class MonthlyTurnSystem(TradingSystem):

    def __init__(
        self,
        cycle_length: int = 21,
        turn_bars: int = 3,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.cycle_length = cycle_length
        self.turn_bars = turn_bars
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

        if len(closes) < self.atr_period + 1:
            return 0

        bar_position = (len(closes) - 1) % self.cycle_length

        if bar_position < self.turn_bars or bar_position >= self.cycle_length - self.turn_bars:
            return 1

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
        bar_position = (len(closes) - 1) % self.cycle_length if len(closes) > 0 else None
        is_turn = (bar_position is not None and
                   (bar_position < self.turn_bars or
                    bar_position >= self.cycle_length - self.turn_bars))
        return {"bar_position": bar_position, "is_turn_of_month": is_turn}
