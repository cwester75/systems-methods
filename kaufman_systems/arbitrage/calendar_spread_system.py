"""
Calendar Spread System

Concept
-------
Near-term vs far-term momentum divergence.  Compares the
return over a short (near) period to the return over a longer
(far) period.  When the near return exceeds the far return by
more than a threshold the market is accelerating (LONG); when
it lags, the market is decelerating (SHORT).

Kaufman discusses calendar spreads and the information in
near-vs-far contract divergence.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CalendarSpreadSystem(TradingSystem):

    def __init__(
        self,
        near_period: int = 5,
        far_period: int = 20,
        threshold: float = 0.02,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.near_period = near_period
        self.far_period = far_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _spread(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.far_period + 1:
            return None
        price = closes[-1]
        near_ref = closes[-(self.near_period + 1)]
        far_ref = closes[-(self.far_period + 1)]
        if near_ref == 0 or far_ref == 0:
            return None
        near_ret = (price - near_ref) / near_ref
        far_ret = (price - far_ref) / far_ref
        return near_ret - far_ret

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
        spread = self._spread(data["closes"])
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
        spread = self._spread(data["closes"])
        return {"calendar_spread": spread}
