"""
Triple Screen System

Based on Alexander Elder's Triple Screen methodology:
  1st screen → trend (long MA)
  2nd screen → oscillator (RSI)
  3rd screen → breakout trigger (short-term high/low)

Corresponds to Kaufman Chapter 19 — Multiple Time Frames.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class TripleScreenSystem(TradingSystem):

    def __init__(
        self,
        trend_period: int = 50,
        rsi_period: int = 14,
        breakout_period: int = 5,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        self.breakout_period = breakout_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def rsi(self, closes):
        closes = np.asarray(closes)

        if len(closes) < self.rsi_period + 1:
            return None

        deltas = np.diff(closes[-(self.rsi_period + 1) :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def signal(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])

        if len(closes) < self.trend_period:
            return 0

        # Screen 1: Trend
        ma = np.mean(closes[-self.trend_period :])
        trend = 1 if closes[-1] > ma else -1

        # Screen 2: Oscillator
        rsi_val = self.rsi(closes)
        if rsi_val is None:
            return 0

        # Screen 3: Breakout trigger
        if len(highs) < self.breakout_period:
            return 0

        recent_high = np.max(highs[-self.breakout_period :])
        recent_low = np.min(np.asarray(data["lows"])[-self.breakout_period :])

        if trend > 0 and rsi_val < self.rsi_oversold and closes[-1] >= recent_high:
            return 1

        if trend < 0 and rsi_val > self.rsi_overbought and closes[-1] <= recent_low:
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
        return len(closes) >= self.trend_period

    def indicators(self, data):
        closes = np.asarray(data["closes"])

        ma = None
        if len(closes) >= self.trend_period:
            ma = float(np.mean(closes[-self.trend_period :]))

        return {
            "trend_ma": ma,
            "rsi": self.rsi(closes),
        }
