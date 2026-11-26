"""
Holiday Effect Pattern System

Concept
-------
Pre-holiday periods often exhibit low volatility and bullish drift.
This system uses volatility contraction as a holiday proxy: when
short-term ATR drops below a fraction of longer-term ATR, the
market is in a low-volatility "holiday-like" condition.

Signal logic
------------
short_atr = ATR over short_atr_period bars
long_atr  = ATR over long_atr_period bars

If short_atr < contraction_pct * long_atr -> LONG (bullish holiday effect)
Otherwise                                 -> FLAT

Risk model
----------
ATR-based position sizing (uses long_atr for sizing).
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class HolidayEffectSystem(TradingSystem):

    def __init__(
        self,
        short_atr_period: int = 5,
        long_atr_period: int = 20,
        contraction_pct: float = 0.6,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.short_atr_period = short_atr_period
        self.long_atr_period = long_atr_period
        self.contraction_pct = contraction_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _compute_atr(self, highs, lows, closes, period):
        highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
        if len(closes) < period + 1:
            return None
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-period:])

    def atr(self, highs, lows, closes):
        return self._compute_atr(highs, lows, closes, self.atr_period)

    def signal(self, data):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]

        short_atr = self._compute_atr(highs, lows, closes, self.short_atr_period)
        long_atr = self._compute_atr(highs, lows, closes, self.long_atr_period)

        if short_atr is None or long_atr is None or long_atr == 0:
            return 0

        if short_atr < self.contraction_pct * long_atr:
            return 1

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
        short_atr = self._compute_atr(highs, lows, closes, self.short_atr_period)
        long_atr = self._compute_atr(highs, lows, closes, self.long_atr_period)
        ratio = short_atr / long_atr if short_atr is not None and long_atr and long_atr != 0 else None
        is_contraction = ratio is not None and ratio < self.contraction_pct
        return {"short_atr": short_atr, "long_atr": long_atr, "atr_ratio": ratio, "is_contraction": is_contraction}
