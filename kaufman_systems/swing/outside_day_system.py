"""
Outside Day System

Concept
-------
An outside day (engulfing bar) occurs when the current bar's high
exceeds the previous bar's high AND the current bar's low is below the
previous bar's low.  The close direction determines the signal.

Signal logic
------------
Outside day + close > previous close → LONG
Outside day + close < previous close → SHORT
Otherwise                            → FLAT

Outside days represent a sudden expansion in volatility and often mark
short-term turning points or continuation accelerations.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OutsideDaySystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def is_outside_day(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < 2:
            return False

        return highs[-1] > highs[-2] and lows[-1] < lows[-2]

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

        return np.mean(tr[-self.atr_period:])

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        closes = np.asarray(closes)

        if len(closes) < 2:
            return 0

        if not self.is_outside_day(highs, lows):
            return 0

        if closes[-1] > closes[-2]:
            return 1

        if closes[-1] < closes[-2]:
            return -1

        return 0

    def position_sizing(self, data, risk):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None or atr <= 0:
            return False

        return True

    def indicators(self, data):
        highs = data["highs"]
        lows = data["lows"]

        return {
            "is_outside_day": self.is_outside_day(highs, lows),
        }
