"""
Inter-Exchange Arbitrage System

Concept
-------
Price deviation from a channel midpoint as fair value.
The midpoint is (SMA(highs) + SMA(lows)) / 2.  When the
close falls below the midpoint by more than threshold ATRs
the price is cheap (LONG); above it the price is rich (SHORT).

Kaufman discusses inter-exchange arbitrage where price
discrepancies between venues are exploited.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class InterexchangeArbitrageSystem(TradingSystem):

    def __init__(
        self,
        channel_period: int = 30,
        threshold: float = 1.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.channel_period = channel_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _midpoint(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        if len(highs) < self.channel_period:
            return None
        h_ma = np.mean(highs[-self.channel_period:])
        l_ma = np.mean(lows[-self.channel_period:])
        return (h_ma + l_ma) / 2

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
        mid = self._midpoint(data["highs"], data["lows"])
        if mid is None:
            return 0
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val == 0:
            return 0
        closes = np.asarray(data["closes"])
        deviation = (closes[-1] - mid) / atr_val
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
        mid = self._midpoint(data["highs"], data["lows"])
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        deviation = None
        if mid is not None and atr_val is not None and atr_val != 0:
            closes = np.asarray(data["closes"])
            deviation = (closes[-1] - mid) / atr_val
        return {"midpoint": mid, "atr_deviation": deviation}
