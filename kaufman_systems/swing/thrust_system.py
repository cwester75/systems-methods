"""
Thrust System

Concept
-------
A thrust is a strong directional move measured as the percentage
change over N bars.  When the move exceeds a threshold, a trend
signal fires in the direction of the thrust.

Signal logic
------------
ROC(N) >  threshold → LONG
ROC(N) < -threshold → SHORT
Otherwise           → FLAT

Kaufman describes thrust methods as confirmation that a market has
enough momentum to sustain a directional move.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ThrustSystem(TradingSystem):

    def __init__(
        self,
        thrust_period: int = 5,
        threshold: float = 0.03,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.thrust_period = thrust_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def thrust(self, closes):
        closes = np.asarray(closes)

        if len(closes) < self.thrust_period + 1:
            return None

        prev = closes[-(self.thrust_period + 1)]

        if prev == 0:
            return None

        return (closes[-1] - prev) / prev

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

        t = self.thrust(closes)

        if t is None:
            return 0

        if t > self.threshold:
            return 1

        if t < -self.threshold:
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
            "thrust": self.thrust(closes),
            "threshold": self.threshold,
        }
