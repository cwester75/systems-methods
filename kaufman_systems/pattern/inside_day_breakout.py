"""
Inside Day Breakout System

Concept
-------
An inside day occurs when the current bar's range is entirely
contained within the previous bar's range (lower high, higher low).
This signals compression.  A breakout from the mother bar's range
on the next bar triggers a directional signal.

Signal logic
------------
If the previous bar was an inside day (bar[-2] contained within bar[-3]):
  Close > high of mother bar (bar[-3]) → LONG
  Close < low of mother bar  (bar[-3]) → SHORT
Otherwise                              → FLAT

We look for an inside day at bar[-2] relative to bar[-3] (the "mother"
bar), then check if the current bar[-1] breaks out of the mother bar's
range.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class InsideDayBreakoutSystem(TradingSystem):

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

    def inside_day_setup(self, highs, lows):
        """Check if bar[-2] is inside bar[-3] and return mother bar levels."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < 3:
            return False, None, None

        mother_high = highs[-3]
        mother_low = lows[-3]
        inside_high = highs[-2]
        inside_low = lows[-2]

        is_inside = inside_high <= mother_high and inside_low >= mother_low

        if is_inside:
            return True, mother_high, mother_low

        return False, None, None

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

        is_setup, mother_high, mother_low = self.inside_day_setup(highs, lows)

        if not is_setup:
            return 0

        price = closes[-1]

        if price > mother_high:
            return 1

        if price < mother_low:
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

        is_setup, mother_high, mother_low = self.inside_day_setup(highs, lows)

        return {
            "is_inside_day_setup": is_setup,
            "mother_high": mother_high,
            "mother_low": mother_low,
        }
