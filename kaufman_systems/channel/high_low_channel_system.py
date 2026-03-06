"""
High-Low Channel System

Concept
-------
Channel defined by separate moving averages of the highs and lows.
The upper band is the SMA of highs; the lower band is the SMA of lows.
This creates a natural volatility-adaptive envelope around price.

Signal logic
------------
Close > SMA(Highs, N) → LONG
Close < SMA(Lows, N)  → SHORT
Otherwise             → FLAT

Kaufman describes high-low channels as a structural way to define
support and resistance without arbitrary band offsets.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class HighLowChannelSystem(TradingSystem):

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def high_channel(self, highs):
        highs = np.asarray(highs)

        if len(highs) < self.channel_period:
            return None

        return np.mean(highs[-self.channel_period:])

    def low_channel(self, lows):
        lows = np.asarray(lows)

        if len(lows) < self.channel_period:
            return None

        return np.mean(lows[-self.channel_period:])

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

        upper = self.high_channel(highs)
        lower = self.low_channel(lows)

        if upper is None or lower is None:
            return 0

        price = closes[-1]

        if price > upper:
            return 1

        if price < lower:
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
            "high_channel": self.high_channel(highs),
            "low_channel": self.low_channel(lows),
        }
