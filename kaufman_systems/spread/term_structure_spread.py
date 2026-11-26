"""
Term Structure Spread System

Concept
-------
Term structure of momentum: fast momentum minus slow momentum.
When the fast window's return exceeds the slow window's by more
than a threshold, the market is in contango-like acceleration
(LONG).  When it lags, the market is in backwardation-like
deceleration (SHORT).

Kaufman discusses near-far contract spreads and term-structure
dynamics.  This single-instrument version captures the same
momentum-divergence logic.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class TermStructureSpreadSystem(TradingSystem):

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        threshold: float = 0.01,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _momentum_spread(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.slow_period + 1:
            return None
        price = closes[-1]
        fast_ref = closes[-(self.fast_period + 1)]
        slow_ref = closes[-(self.slow_period + 1)]
        if fast_ref == 0 or slow_ref == 0:
            return None
        fast_mom = (price - fast_ref) / fast_ref
        slow_mom = (price - slow_ref) / slow_ref
        return fast_mom - slow_mom

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
        spread = self._momentum_spread(data["closes"])
        if spread is None:
            return 0
        if spread > self.threshold:
            return 1
        if spread < -self.threshold:
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
        spread = self._momentum_spread(data["closes"])
        return {"momentum_spread": spread}
