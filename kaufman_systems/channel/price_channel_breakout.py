"""
Price Channel Breakout System

Concept
-------
Classic N-period price channel breakout.  The channel is defined by the
highest high and lowest low over a lookback window, excluding the
current bar.  A breakout occurs when the current close exceeds the
channel boundary.

Signal logic
------------
Close > Highest High (lookback, excluding current bar) → LONG
Close < Lowest Low   (lookback, excluding current bar) → SHORT
Otherwise                                              → FLAT

This differs from the Donchian system by explicitly excluding the
current bar from the channel calculation, giving a cleaner breakout
signal on the bar that actually breaks the level.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PriceChannelBreakoutSystem(TradingSystem):

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

    def channel_high(self, highs):
        highs = np.asarray(highs)

        if len(highs) < self.channel_period + 1:
            return None

        return np.max(highs[-(self.channel_period + 1):-1])

    def channel_low(self, lows):
        lows = np.asarray(lows)

        if len(lows) < self.channel_period + 1:
            return None

        return np.min(lows[-(self.channel_period + 1):-1])

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

        ch_high = self.channel_high(highs)
        ch_low = self.channel_low(lows)

        if ch_high is None or ch_low is None:
            return 0

        price = closes[-1]

        if price > ch_high:
            return 1

        if price < ch_low:
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
            "channel_high": self.channel_high(highs),
            "channel_low": self.channel_low(lows),
        }
