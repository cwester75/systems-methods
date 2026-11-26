"""
Carry Spread System

Concept
-------
Carry proxy via the linear regression slope of price over a
lookback window, normalized by price.  Positive slope implies
positive carry (LONG); negative slope implies negative carry
(SHORT).  A threshold filters noise.

Kaufman discusses carry differentials in the context of
interest-rate and commodity term structure trades.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CarrySpreadSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 30,
        threshold: float = 0.001,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _normalized_slope(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.lookback:
            return None
        window = closes[-self.lookback:]
        price = window[-1]
        if price == 0:
            return None
        x = np.arange(self.lookback)
        slope = np.polyfit(x, window, 1)[0]
        return slope / price

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
        ns = self._normalized_slope(data["closes"])
        if ns is None:
            return 0
        if ns > self.threshold:
            return 1
        if ns < -self.threshold:
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
        ns = self._normalized_slope(data["closes"])
        return {"normalized_slope": ns}
