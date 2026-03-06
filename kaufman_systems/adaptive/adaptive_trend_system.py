"""
Adaptive Trend System

Adjusts moving average length based on volatility.
High volatility → shorter lookback (faster adaptation)
Low volatility → longer lookback (smoother trend)

Corresponds to Kaufman Chapter 17 — Adaptive Techniques.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class AdaptiveTrendSystem(TradingSystem):

    def __init__(
        self,
        base_period: int = 30,
        min_period: int = 5,
        max_period: int = 100,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.base_period = base_period
        self.min_period = min_period
        self.max_period = max_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def adaptive_period(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.atr_period + 1:
            return self.base_period

        returns = np.abs(np.diff(closes[-(self.atr_period + 1) :]))
        vol = np.mean(returns) / closes[-1]

        adj = int(self.base_period * (1 + vol * 100))
        return max(self.min_period, min(adj, self.max_period))

    def signal(self, data):
        closes = np.asarray(data["closes"])
        period = self.adaptive_period(closes)

        if len(closes) < period:
            return 0

        ma = np.mean(closes[-period:])

        if closes[-1] > ma:
            return 1
        elif closes[-1] < ma:
            return -1
        return 0

    def position_sizing(self, data, risk):
        closes = np.asarray(data["closes"])
        equity = risk["equity"]

        if len(closes) < self.atr_period + 1:
            return 0

        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

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
        if len(closes) < self.atr_period + 1:
            return False
        return True

    def indicators(self, data):
        closes = data["closes"]
        return {
            "adaptive_period": self.adaptive_period(closes),
        }
