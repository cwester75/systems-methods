"""
Dual Timeframe Trend System

Higher timeframe determines trend direction.
Lower timeframe provides entry timing.

Example: Weekly MA → trend filter, Daily breakout → entry.

Corresponds to Kaufman Chapter 19 — Multiple Time Frames.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class DualTimeframeTrend(TradingSystem):

    def __init__(
        self,
        slow_period: int = 50,
        fast_period: int = 10,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.slow_period = slow_period
        self.fast_period = fast_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def signal(self, data):
        closes = np.asarray(data["closes"])

        if len(closes) < self.slow_period:
            return 0

        slow_ma = np.mean(closes[-self.slow_period :])
        fast_ma = np.mean(closes[-self.fast_period :])

        trend = 1 if closes[-1] > slow_ma else -1
        entry = 1 if closes[-1] > fast_ma else -1

        if trend > 0 and entry > 0:
            return 1
        if trend < 0 and entry < 0:
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
        return len(closes) >= self.slow_period

    def indicators(self, data):
        closes = np.asarray(data["closes"])

        slow_ma = None
        fast_ma = None

        if len(closes) >= self.slow_period:
            slow_ma = float(np.mean(closes[-self.slow_period :]))
        if len(closes) >= self.fast_period:
            fast_ma = float(np.mean(closes[-self.fast_period :]))

        return {
            "slow_ma": slow_ma,
            "fast_ma": fast_ma,
        }
