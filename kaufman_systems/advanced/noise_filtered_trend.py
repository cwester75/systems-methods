"""
Noise Filtered Trend System

Trades only when the Efficiency Ratio exceeds a threshold,
filtering out noisy, trendless markets.

Corresponds to Kaufman Chapter 20 — Advanced Techniques.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class NoiseFilteredTrend(TradingSystem):

    def __init__(
        self,
        er_period: int = 10,
        ma_period: int = 20,
        er_threshold: float = 0.3,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.er_period = er_period
        self.ma_period = ma_period
        self.er_threshold = er_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def efficiency_ratio(self, closes):
        closes = np.asarray(closes)

        if len(closes) < self.er_period + 1:
            return None

        change = abs(closes[-1] - closes[-1 - self.er_period])
        volatility = np.sum(np.abs(np.diff(closes[-1 - self.er_period :])))

        if volatility == 0:
            return 0

        return change / volatility

    def signal(self, data):
        closes = np.asarray(data["closes"])

        er = self.efficiency_ratio(closes)
        if er is None or er < self.er_threshold:
            return 0

        if len(closes) < self.ma_period:
            return 0

        ma = np.mean(closes[-self.ma_period :])

        if closes[-1] > ma:
            return 1
        elif closes[-1] < ma:
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
        er = self.efficiency_ratio(closes)
        return er is not None and er >= self.er_threshold

    def indicators(self, data):
        closes = data["closes"]
        return {
            "efficiency_ratio": self.efficiency_ratio(closes),
        }
