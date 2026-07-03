"""
Narrow Range Breakout System (NR4/NR7)

Concept
-------
Identifies bars with the narrowest range in N periods (classic NR4 or
NR7 patterns).  A narrow-range bar signals compression; a breakout
from that bar's high or low on the next bar triggers a signal.

Signal logic
------------
If bar[-2] had the narrowest range in the last N bars:
  Close > high of bar[-2] → LONG
  Close < low of bar[-2]  → SHORT
Otherwise                 → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class NarrowRangeBreakoutSystem(TradingSystem):

    def __init__(
        self,
        nr_period: int = 7,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.nr_period = nr_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def is_narrow_range(self, highs, lows):
        """Check if bar[-2] has the narrowest range in the last nr_period bars."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.nr_period + 1:
            return False, None, None

        ranges = highs - lows
        lookback_ranges = ranges[-(self.nr_period + 1):-1]
        nr_bar_range = ranges[-2]

        if nr_bar_range == np.min(lookback_ranges):
            return True, highs[-2], lows[-2]

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

        is_nr, nr_high, nr_low = self.is_narrow_range(highs, lows)

        if not is_nr:
            return 0

        price = closes[-1]

        if price > nr_high:
            return 1

        if price < nr_low:
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

        is_nr, nr_high, nr_low = self.is_narrow_range(highs, lows)

        return {
            "is_narrow_range": is_nr,
            "nr_high": nr_high,
            "nr_low": nr_low,
        }
