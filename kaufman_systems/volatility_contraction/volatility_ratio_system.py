"""
Volatility Ratio System

Concept
-------
The volatility ratio compares the current bar's true range to the
average true range.  A ratio significantly above 1.0 indicates a
volatility expansion — a potential breakout.  The close direction
determines the signal.

Signal logic
------------
If TR / ATR > expansion_threshold:
  Close > prev close → LONG
  Close < prev close → SHORT
Otherwise            → FLAT

Kaufman uses the volatility ratio as a filter to identify bars where
price movement is unusually large relative to recent norms.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class VolatilityRatioSystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        expansion_threshold: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.expansion_threshold = expansion_threshold
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def true_range(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(closes) < 2:
            return None

        return max(
            highs[-1] - lows[-1],
            abs(highs[-1] - closes[-2]),
            abs(lows[-1] - closes[-2]),
        )

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

        tr = self.true_range(highs, lows, closes)
        atr = self.atr(highs, lows, closes)

        if tr is None or atr is None or atr == 0:
            return 0

        ratio = tr / atr

        if ratio <= self.expansion_threshold:
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
        closes = data["closes"]

        tr = self.true_range(highs, lows, closes)
        atr = self.atr(highs, lows, closes)

        ratio = tr / atr if tr is not None and atr is not None and atr != 0 else None

        return {
            "true_range": tr,
            "atr": atr,
            "volatility_ratio": ratio,
        }
