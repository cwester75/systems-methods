"""
Cash and Carry System

Concept
-------
Fair value approximated by the SMA.  When the close deviates
below the SMA by more than threshold ATRs the price is cheap
relative to its "carry" value (LONG); above it the price is
rich (SHORT).

Kaufman discusses cash-and-carry arbitrage where the futures
price is compared to the spot price plus carrying costs.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CashAndCarrySystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 50,
        threshold: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.threshold = threshold
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
        if len(closes) < self.ma_period:
            return 0
        atr_val = self.atr(data["highs"], data["lows"], closes)
        if atr_val is None or atr_val == 0:
            return 0
        ma = np.mean(closes[-self.ma_period:])
        deviation = (closes[-1] - ma) / atr_val
        if deviation < -self.threshold:
            return 1
        if deviation > self.threshold:
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
        closes = np.asarray(data["closes"])
        atr_val = self.atr(data["highs"], data["lows"], closes)
        ma = np.mean(closes[-self.ma_period:]) if len(closes) >= self.ma_period else None
        deviation = None
        if ma is not None and atr_val is not None and atr_val != 0:
            deviation = (closes[-1] - ma) / atr_val
        return {"fair_value": ma, "atr_deviation": deviation}
