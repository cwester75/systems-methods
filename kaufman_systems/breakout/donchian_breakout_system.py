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

from kaufman_systems.base import TradingSystem


class DonchianBreakoutSystem(TradingSystem):

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

    def signal(self, data):
        """
        Returns
        -------
        1  → Long breakout
       -1  → Short breakout
        0  → No signal
        """

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

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

    def position_sizing(self, data, risk):
        """
        ATR-based risk sizing.
        """

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, data):
        """
        Basic volatility sanity check.
        """

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, data):

        highs = data["highs"]
        lows = data["lows"]

        return {
            "donchian_high": self.channel_high(highs),
            "donchian_low": self.channel_low(lows),
        }
