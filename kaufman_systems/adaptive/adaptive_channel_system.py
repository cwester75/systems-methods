"""
Adaptive Channel System

Channel width adapts dynamically based on ATR.
Breakouts above/below the volatility-normalized channel generate signals.

Corresponds to Kaufman Chapter 17 — Adaptive Techniques.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class AdaptiveChannelSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        atr_period: int = 14,
        channel_mult: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.channel_mult = channel_mult
        self.risk_per_trade = risk_per_trade

    def atr(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(closes) < self.atr_period + 1:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-self.atr_period :])

    def signal(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        if len(closes) < max(self.ma_period, self.atr_period + 1):
            return 0

        ma = np.mean(closes[-self.ma_period :])
        atr_val = self.atr(highs, lows, closes)

        if atr_val is None:
            return 0

        upper = ma + self.channel_mult * atr_val
        lower = ma - self.channel_mult * atr_val

        if closes[-1] > upper:
            return 1
        elif closes[-1] < lower:
            return -1
        return 0

    def position_sizing(self, data, risk):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])
        equity = risk["equity"]

        atr_val = self.atr(highs, lows, closes)

        if atr_val is None or atr_val == 0:
            return 0

        return equity * self.risk_per_trade / atr_val

    def risk_filter(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        atr_val = self.atr(highs, lows, closes)

        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        atr_val = self.atr(highs, lows, closes)
        ma = None
        upper = None
        lower = None

        if len(closes) >= self.ma_period and atr_val is not None:
            ma = float(np.mean(closes[-self.ma_period :]))
            upper = ma + self.channel_mult * atr_val
            lower = ma - self.channel_mult * atr_val

        return {
            "atr": atr_val,
            "ma": ma,
            "upper": upper,
            "lower": lower,
        }
