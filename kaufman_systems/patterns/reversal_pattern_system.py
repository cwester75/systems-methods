"""
Reversal Pattern System

Concept
-------
Reversal bars indicate exhaustion of the current move.

Bullish Key Reversal
    Bar makes a new N-bar low but closes above the prior close.

Bearish Key Reversal
    Bar makes a new N-bar high but closes below the prior close.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ReversalPatternSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

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

    def _bullish_reversal(self, highs, lows, closes):
        lows = np.asarray(lows)
        closes = np.asarray(closes)
        if len(closes) < self.lookback + 1:
            return False
        lowest = np.min(lows[-(self.lookback + 1):-1])
        return bool(lows[-1] < lowest and closes[-1] > closes[-2])

    def _bearish_reversal(self, highs, lows, closes):
        highs = np.asarray(highs)
        closes = np.asarray(closes)
        if len(closes) < self.lookback + 1:
            return False
        highest = np.max(highs[-(self.lookback + 1):-1])
        return bool(highs[-1] > highest and closes[-1] < closes[-2])

    def signal(self, data):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]
        if self._bullish_reversal(highs, lows, closes):
            return 1
        if self._bearish_reversal(highs, lows, closes):
            return -1
        return 0

    def position_sizing(self, data, risk):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return 0
        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]
        return {
            "bullish_reversal": self._bullish_reversal(highs, lows, closes),
            "bearish_reversal": self._bearish_reversal(highs, lows, closes),
        }
