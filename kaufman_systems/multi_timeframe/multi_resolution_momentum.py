"""
Multi-Resolution Momentum System

Measures momentum across multiple horizons (short, medium, long).
Trades when momentum aligns across all scales.

composite_signal = sign(M_short) + sign(M_medium) + sign(M_long)

Corresponds to Kaufman Chapter 19 — Multiple Time Frames.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class MultiResolutionMomentum(TradingSystem):

    def __init__(
        self,
        short_period: int = 5,
        medium_period: int = 20,
        long_period: int = 60,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def signal(self, data):
        closes = np.asarray(data["closes"])

        if len(closes) < self.long_period + 1:
            return 0

        m_short = closes[-1] - closes[-1 - self.short_period]
        m_medium = closes[-1] - closes[-1 - self.medium_period]
        m_long = closes[-1] - closes[-1 - self.long_period]

        composite = int(np.sign(m_short) + np.sign(m_medium) + np.sign(m_long))

        if composite == 3:
            return 1
        elif composite == -3:
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
        return len(closes) >= self.long_period + 1

    def indicators(self, data):
        closes = np.asarray(data["closes"])

        if len(closes) < self.long_period + 1:
            return {"m_short": None, "m_medium": None, "m_long": None}

        return {
            "m_short": float(closes[-1] - closes[-1 - self.short_period]),
            "m_medium": float(closes[-1] - closes[-1 - self.medium_period]),
            "m_long": float(closes[-1] - closes[-1 - self.long_period]),
        }
