"""
Range Percentile System

Concept
-------
Ranks the current bar's range against the distribution of ranges over
the last N bars.  When the current range falls in an extreme percentile,
it signals either compression (ready to expand) or expansion (breakout
in progress).

Signal logic
------------
If range percentile > upper_pct (e.g., 90th percentile — expansion):
  Close > prev close → LONG
  Close < prev close → SHORT
Otherwise            → FLAT

Kaufman discusses percentile-based filters as a way to rank current
volatility relative to recent history.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RangePercentileSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 50,
        upper_pct: float = 90.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.upper_pct = upper_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def range_percentile(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.lookback + 1:
            return None

        ranges = highs - lows
        current_range = ranges[-1]
        historical_ranges = ranges[-(self.lookback + 1):-1]

        percentile = np.sum(historical_ranges <= current_range) / len(historical_ranges) * 100

        return percentile

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

        pct = self.range_percentile(highs, lows)

        if pct is None:
            return 0

        if pct <= self.upper_pct:
            return 0

        if len(closes) < 2:
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
            "range_percentile": self.range_percentile(highs, lows),
            "upper_pct": self.upper_pct,
        }
