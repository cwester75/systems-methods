"""
Intermarket Spread System

Concept
-------
Z-score of close relative to its long-term SMA, used as a
single-instrument spread proxy.  When the z-score falls below
-entry_z the market is undervalued relative to trend (LONG);
above +entry_z it is overvalued (SHORT).

Kaufman discusses intermarket relationships and mean-reversion
across related instruments.  With a single price series the
z-score captures the same reversion logic.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class IntermarketSpreadSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 50,
        entry_z: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _zscore(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.lookback:
            return None
        window = closes[-self.lookback:]
        ma = np.mean(window)
        sd = np.std(window)
        if sd == 0:
            return None
        return (closes[-1] - ma) / sd

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
        z = self._zscore(data["closes"])
        if z is None:
            return 0
        if z < -self.entry_z:
            return 1
        if z > self.entry_z:
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
        z = self._zscore(data["closes"])
        return {"zscore": z}
