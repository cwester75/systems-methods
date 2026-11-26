"""
Opinion Indicator System

Concept
-------
Uses consecutive up/down close days as a proxy for crowd opinion.
A long streak of consecutive gains suggests the crowd is unanimously
bullish — a contrarian signal to sell. Vice versa for down streaks.

Signal logic
------------
Count consecutive bars where close > prev close (or close < prev close).
If >= threshold consecutive up days → crowd bullish → SHORT (contrarian)
If >= threshold consecutive down days → crowd bearish → LONG (contrarian)
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OpinionIndicatorSystem(TradingSystem):

    def __init__(
        self,
        streak_threshold: int = 5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.streak_threshold = streak_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _streak(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.streak_threshold + 1:
            return 0
        diffs = np.diff(closes)
        count = 0
        direction = 0
        for d in reversed(diffs):
            if d > 0:
                if direction == 0:
                    direction = 1
                if direction == 1:
                    count += 1
                else:
                    break
            elif d < 0:
                if direction == 0:
                    direction = -1
                if direction == -1:
                    count += 1
                else:
                    break
            else:
                break
        return count * direction

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
        closes = data["closes"]
        if len(np.asarray(closes)) < self.streak_threshold + 1:
            return 0
        streak = self._streak(closes)
        if streak >= self.streak_threshold:
            return -1
        if streak <= -self.streak_threshold:
            return 1
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
        streak = self._streak(data["closes"])
        return {
            "streak": streak,
            "is_extreme_bullish": streak >= self.streak_threshold,
            "is_extreme_bearish": streak <= -self.streak_threshold,
        }
