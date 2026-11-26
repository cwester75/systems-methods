"""
Mean Reversion Spread System

Concept
-------
Deviation of close from its N-bar SMA, normalized by ATR.
When the deviation exceeds +threshold ATRs the price is stretched
above its mean (SHORT); below -threshold it is compressed (LONG).

Kaufman discusses mean-reversion strategies that exploit the
tendency of prices to return to their average value.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class MeanReversionSpreadSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 30,
        threshold: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _deviation(self, closes, highs, lows):
        closes = np.asarray(closes)
        if len(closes) < self.ma_period:
            return None
        ma = np.mean(closes[-self.ma_period:])
        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return None
        return (closes[-1] - ma) / atr_val

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
        dev = self._deviation(data["closes"], data["highs"], data["lows"])
        if dev is None:
            return 0
        if dev < -self.threshold:
            return 1
        if dev > self.threshold:
            return -1
        return 0

    def position_sizing(self, data, risk):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val == 0:
            return 0
        equity = risk["equity"]
        rpt = risk.get("risk_per_trade", self.risk_per_trade)
        return equity * rpt / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        dev = self._deviation(data["closes"], data["highs"], data["lows"])
        return {"atr_deviation": dev}
