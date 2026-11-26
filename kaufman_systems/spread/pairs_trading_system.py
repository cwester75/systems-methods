"""
Pairs Trading System (Self-Regression)

Concept
-------
Self-regression z-score: fit a linear regression of the close
on its own lagged values and compute the z-score of the residual.
When the z-score is below -entry_z the price is cheap relative
to its own history (LONG); above +entry_z it is rich (SHORT).

This captures the cointegration / mean-reversion logic Kaufman
describes for pairs and spread trading, applied to a single
instrument's autoregressive structure.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PairsTradingSystem(TradingSystem):

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

    def _residual_zscore(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.lookback + 1:
            return None
        window = closes[-(self.lookback + 1):]
        y = window[1:]
        x = window[:-1]
        coeffs = np.polyfit(x, y, 1)
        predicted = np.polyval(coeffs, x)
        residuals = y - predicted
        sd = np.std(residuals)
        if sd == 0:
            return None
        return residuals[-1] / sd

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
        z = self._residual_zscore(data["closes"])
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
        z = self._residual_zscore(data["closes"])
        return {"residual_zscore": z}
