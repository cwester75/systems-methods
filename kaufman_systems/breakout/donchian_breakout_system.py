"""
Donchian Channel Breakout System

Concept
-------
Classic trend-following breakout system popularized by the Turtle traders.

Signal logic
------------
Price > Highest High (N)  → LONG
Price < Lowest Low (N)    → SHORT
Otherwise                 → FLAT

The channel represents the rolling price extremes over N periods.

Position sizing
---------------
ATR-based risk normalization to standardize exposure across instruments.
"""

import numpy as np


class DonchianBreakoutSystem:

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
    # Donchian Channel
    # ---------------------------------------------------------

    def channel_high(self, highs):
        highs = np.asarray(highs)

        if len(highs) < self.channel_period:
            return None

        return np.max(highs[-self.channel_period:])

    def channel_low(self, lows):
        lows = np.asarray(lows)

        if len(lows) < self.channel_period:
            return None

        return np.min(lows[-self.channel_period:])

    # ---------------------------------------------------------
    # ATR Calculation
    # ---------------------------------------------------------

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
    # Core System Interface
    # ---------------------------------------------------------

    def signal(self, highs, lows, closes):
        """
        Returns
        -------
        1  → Long breakout
       -1  → Short breakout
        0  → No signal
        """

        high_channel = self.channel_high(highs)
        low_channel = self.channel_low(lows)

        if high_channel is None or low_channel is None:
            return 0

        price = closes[-1]

        if price > high_channel:
            return 1

        if price < low_channel:
            return -1

        return 0

    def position_sizing(self, equity, highs, lows, closes):
        """
        ATR-based risk sizing.
        """

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * self.risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, highs, lows, closes):
        """
        Basic volatility sanity check.
        """

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, highs, lows):

        return {
            "donchian_high": self.channel_high(highs),
            "donchian_low": self.channel_low(lows),
        }
