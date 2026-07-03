"""
Standard Deviation Breakout System

Concept
-------
Uses the rolling standard deviation of closes as a volatility measure.
When price moves beyond k standard deviations from the mean, a breakout
signal fires.  Similar to Bollinger Bands but framed as a pure
statistical Z-score breakout.

Signal logic
------------
Z-score = (Close - SMA) / StdDev
If Z-score >  threshold → LONG
If Z-score < -threshold → SHORT
Otherwise               → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class StandardDeviationBreakoutSystem(TradingSystem):

    def __init__(
        self,
        window: int = 20,
        threshold: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.window = window
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def zscore(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.window:
            return None

        window = prices[-self.window:]
        ma = np.mean(window)
        sd = np.std(window)

        if sd == 0:
            return 0.0

        return (prices[-1] - ma) / sd

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
        closes = data["closes"]

        z = self.zscore(closes)

        if z is None:
            return 0

        if z > self.threshold:
            return 1

        if z < -self.threshold:
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
        closes = data["closes"]

        return {
            "zscore": self.zscore(closes),
            "threshold": self.threshold,
        }
