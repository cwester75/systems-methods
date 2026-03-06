"""
Volatility Weighted Trend System

Trend signals weighted by volatility.
Trade only when trend_strength = |slope| / volatility exceeds a threshold.

Corresponds to Kaufman Chapter 20 — Advanced Techniques.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class VolatilityWeightedTrend(TradingSystem):

    def __init__(
        self,
        trend_period: int = 20,
        vol_period: int = 20,
        strength_threshold: float = 1.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.trend_period = trend_period
        self.vol_period = vol_period
        self.strength_threshold = strength_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def trend_strength(self, closes):
        closes = np.asarray(closes)

        if len(closes) < max(self.trend_period, self.vol_period) + 1:
            return None

        slope = (closes[-1] - closes[-1 - self.trend_period]) / self.trend_period

        returns = np.diff(closes[-(self.vol_period + 1) :]) / closes[-(self.vol_period + 1) : -1]
        vol = np.std(returns)

        if vol == 0:
            return None

        return slope / (vol * closes[-1])

    def signal(self, data):
        closes = np.asarray(data["closes"])
        strength = self.trend_strength(closes)

        if strength is None:
            return 0

        if strength > self.strength_threshold:
            return 1
        elif strength < -self.strength_threshold:
            return -1
        return 0

    def position_sizing(self, data, risk):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])
        equity = risk["equity"]

        if len(closes) < self.atr_period + 1:
            return 0

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = np.mean(tr[-self.atr_period :])

        if atr == 0:
            return 0

        return equity * self.risk_per_trade / atr

    def risk_filter(self, data):
        closes = np.asarray(data["closes"])
        strength = self.trend_strength(closes)
        return strength is not None

    def indicators(self, data):
        closes = data["closes"]
        return {
            "trend_strength": self.trend_strength(closes),
        }
