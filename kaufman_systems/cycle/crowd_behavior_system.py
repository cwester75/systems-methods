"""
Crowd Behavior System

Concept
-------
Herding proxy from price clustering. Measures how tightly closes
cluster around their SMA (std of deviations normalized by ATR).
Low dispersion indicates herding. When herding is detected and price
breaks above/below the SMA, a directional signal fires.

Signal logic
------------
dispersion = std(closes[-N:] - SMA) / ATR
If dispersion < herding_threshold:
  close > SMA → LONG
  close < SMA → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CrowdBehaviorSystem(TradingSystem):

    def __init__(
        self,
        cluster_period: int = 20,
        herding_threshold: float = 0.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.cluster_period = cluster_period
        self.herding_threshold = herding_threshold
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

    def _dispersion(self, data):
        closes = np.asarray(data["closes"])
        min_needed = max(self.cluster_period, self.atr_period + 1)
        if len(closes) < min_needed:
            return None, None
        window = closes[-self.cluster_period:]
        ma = np.mean(window)
        deviations = window - ma
        disp_std = np.std(deviations)
        atr_val = self.atr(data["highs"], data["lows"], closes)
        if atr_val is None or atr_val == 0:
            return None, ma
        return disp_std / atr_val, ma

    def signal(self, data):
        dispersion, ma = self._dispersion(data)
        if dispersion is None:
            return 0
        if dispersion >= self.herding_threshold:
            return 0
        price = np.asarray(data["closes"])[-1]
        if price > ma:
            return 1
        if price < ma:
            return -1
        return 0

    def position_sizing(self, data, risk):
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val == 0:
            return 0
        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        dispersion, ma = self._dispersion(data)
        return {
            "dispersion": dispersion,
            "ma": ma,
            "is_herding": dispersion is not None and dispersion < self.herding_threshold,
        }
