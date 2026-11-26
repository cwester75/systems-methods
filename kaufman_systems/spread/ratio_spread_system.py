"""
Ratio Spread System

Concept
-------
Ratio of a fast SMA to a slow SMA.  When the ratio exceeds
1 + threshold the short-term trend outpaces the long-term (LONG);
when below 1 - threshold the short-term lags (SHORT).

This captures the relative-value divergence between time horizons
that Kaufman describes in spread and ratio analysis.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RatioSpreadSystem(TradingSystem):

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        threshold: float = 0.02,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _ratio(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.slow_period:
            return None
        fast_ma = np.mean(closes[-self.fast_period:])
        slow_ma = np.mean(closes[-self.slow_period:])
        if slow_ma == 0:
            return None
        return fast_ma / slow_ma

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
        ratio = self._ratio(data["closes"])
        if ratio is None:
            return 0
        if ratio > 1 + self.threshold:
            return 1
        if ratio < 1 - self.threshold:
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
        ratio = self._ratio(data["closes"])
        return {"ratio": ratio}
