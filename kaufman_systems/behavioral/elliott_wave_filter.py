"""
Elliott Wave Filter System

Concept
-------
Wave momentum filter that counts swing points (local highs and lows).
When recent swing lows are ascending (impulse up), signal LONG.
When recent swing highs are descending (impulse down), signal SHORT.

Signal logic
------------
Identify local lows (bar lower than its neighbors within swing_lookback).
If the last min_swings swing lows are ascending → LONG
If the last min_swings swing highs are descending → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ElliottWaveFilterSystem(TradingSystem):

    def __init__(
        self,
        swing_lookback: int = 5,
        min_swings: int = 3,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.swing_lookback = swing_lookback
        self.min_swings = min_swings
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _find_swing_points(self, closes):
        closes = np.asarray(closes)
        n = self.swing_lookback
        swing_lows = []
        swing_highs = []
        for i in range(n, len(closes) - n):
            if closes[i] == np.min(closes[i - n:i + n + 1]):
                swing_lows.append(closes[i])
            if closes[i] == np.max(closes[i - n:i + n + 1]):
                swing_highs.append(closes[i])
        return swing_lows, swing_highs

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
        closes = np.asarray(data["closes"])
        min_needed = 2 * self.swing_lookback + self.min_swings
        if len(closes) < min_needed:
            return 0
        swing_lows, swing_highs = self._find_swing_points(closes)
        if len(swing_lows) >= self.min_swings:
            recent = swing_lows[-self.min_swings:]
            if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                return 1
        if len(swing_highs) >= self.min_swings:
            recent = swing_highs[-self.min_swings:]
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
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
        closes = np.asarray(data["closes"])
        swing_lows, swing_highs = [], []
        min_needed = 2 * self.swing_lookback + self.min_swings
        if len(closes) >= min_needed:
            swing_lows, swing_highs = self._find_swing_points(closes)
        return {
            "swing_lows": swing_lows[-self.min_swings:] if len(swing_lows) >= self.min_swings else swing_lows,
            "swing_highs": swing_highs[-self.min_swings:] if len(swing_highs) >= self.min_swings else swing_highs,
            "num_swing_lows": len(swing_lows),
            "num_swing_highs": len(swing_highs),
        }
