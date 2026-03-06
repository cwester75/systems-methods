"""
Adaptive Momentum System

Momentum period adapts to market noise level.
High volatility → shorter lookback (faster response)
Low volatility → longer lookback (more stable signal)

Corresponds to Kaufman Chapter 17 — Adaptive Techniques.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class AdaptiveMomentumSystem(TradingSystem):

    def __init__(
        self,
        base_period: int = 20,
        min_period: int = 5,
        max_period: int = 60,
        vol_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.base_period = base_period
        self.min_period = min_period
        self.max_period = max_period
        self.vol_period = vol_period
        self.risk_per_trade = risk_per_trade

    def adaptive_period(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.vol_period + 1:
            return self.base_period

        returns = np.diff(closes[-(self.vol_period + 1) :]) / closes[-(self.vol_period + 1) : -1]
        vol = np.std(returns)

        if vol == 0:
            return self.base_period

        # Higher vol → shorter period
        adj = int(self.base_period / (1 + vol * 50))
        return max(self.min_period, min(adj, self.max_period))

    def signal(self, data):
        closes = np.asarray(data["closes"])
        period = self.adaptive_period(closes)

        if len(closes) < period + 1:
            return 0

        momentum = closes[-1] - closes[-1 - period]

        if momentum > 0:
            return 1
        elif momentum < 0:
            return -1
        return 0

    def position_sizing(self, data, risk):
        closes = np.asarray(data["closes"])
        equity = risk["equity"]

        if len(closes) < self.vol_period + 1:
            return 0

        returns = np.diff(closes[-(self.vol_period + 1) :]) / closes[-(self.vol_period + 1) : -1]
        vol = np.std(returns)

        if vol == 0:
            return 0

        return equity * self.risk_per_trade / (vol * closes[-1])

    def risk_filter(self, data):
        closes = np.asarray(data["closes"])
        if len(closes) < self.vol_period + 1:
            return False

        returns = np.diff(closes[-(self.vol_period + 1) :]) / closes[-(self.vol_period + 1) : -1]
        vol = np.std(returns)

        return vol > 0

    def indicators(self, data):
        closes = data["closes"]
        period = self.adaptive_period(closes)
        closes = np.asarray(closes)

        momentum = None
        if len(closes) >= period + 1:
            momentum = float(closes[-1] - closes[-1 - period])

        return {
            "adaptive_period": period,
            "momentum": momentum,
        }
