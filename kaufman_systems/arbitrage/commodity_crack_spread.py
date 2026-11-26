"""
Commodity Crack Spread System

Concept
-------
Range-to-price ratio as a processing-margin proxy.  The average
bar range over N periods divided by the close approximates a
"crack spread" — a measure of how much price movement (volatility)
the market is offering relative to price level.  Percentile-based
thresholds against historical crack values determine signals.

When the crack is above the upper percentile the market is in
high-volatility expansion (LONG); below the lower percentile it
is in low-volatility compression (SHORT).

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CommodityCrackSpreadSystem(TradingSystem):

    def __init__(
        self,
        range_period: int = 20,
        upper_pct: float = 75.0,
        lower_pct: float = 25.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.range_period = range_period
        self.upper_pct = upper_pct
        self.lower_pct = lower_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _crack(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)
        if len(closes) < self.range_period + 1:
            return None, None, None
        ranges = highs - lows
        crack_series = ranges[-self.range_period:] / closes[-self.range_period:]
        current_crack = crack_series[-1]
        upper_thresh = np.percentile(crack_series, self.upper_pct)
        lower_thresh = np.percentile(crack_series, self.lower_pct)
        return current_crack, upper_thresh, lower_thresh

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
        crack, upper, lower = self._crack(
            data["highs"], data["lows"], data["closes"]
        )
        if crack is None:
            return 0
        if crack > upper:
            return 1
        if crack < lower:
            return -1
        return 0

    def position_sizing(self, data, risk):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val == 0:
            return 0
        equity = risk["equity"]
        rpt = risk.get("risk_per_trade", self.risk_per_trade)
        return equity * rpt / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        crack, upper, lower = self._crack(
            data["highs"], data["lows"], data["closes"]
        )
        return {
            "crack": crack,
            "upper_threshold": upper,
            "lower_threshold": lower,
        }
