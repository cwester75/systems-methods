"""
Sentiment Contrarian System

Concept
-------
Uses RSI as a proxy for crowd sentiment extremes. Overbought readings
suggest bullish sentiment has gone too far; oversold readings suggest
bearish sentiment is extreme. Signals are contrarian.

Signal logic
------------
RSI = 100 - 100 / (1 + avg_gain / avg_loss)
If RSI > overbought → too bullish → SHORT
If RSI < oversold   → too bearish → LONG
Otherwise           → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class SentimentContrarianSystem(TradingSystem):

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _rsi(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.rsi_period + 1:
            return None
        diffs = np.diff(closes[-(self.rsi_period + 1):])
        gains = np.maximum(diffs, 0.0)
        losses = np.maximum(-diffs, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

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
        rsi = self._rsi(data["closes"])
        if rsi is None:
            return 0
        if rsi > self.overbought:
            return -1
        if rsi < self.oversold:
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
        rsi = self._rsi(data["closes"])
        return {
            "rsi": rsi,
            "is_overbought": rsi is not None and rsi > self.overbought,
            "is_oversold": rsi is not None and rsi < self.oversold,
        }
