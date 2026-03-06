"""
Keltner Squeeze System

Concept
-------
Detects when Bollinger Bands contract inside Keltner Channels, signaling
extreme volatility compression.  The squeeze fires when the Bollinger
upper band drops below the Keltner upper channel and the Bollinger lower
band rises above the Keltner lower channel.  Breakout direction on
release determines the signal.

Signal logic
------------
If Bollinger bands inside Keltner channels (squeeze active):
  Close > Keltner upper → LONG
  Close < Keltner lower → SHORT
Otherwise               → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class KeltnerSqueezeSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        bb_mult: float = 2.0,
        kc_mult: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def bollinger_bands(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.ma_period:
            return None, None, None

        window = prices[-self.ma_period:]
        ma = np.mean(window)
        sd = np.std(window)

        return ma, ma + self.bb_mult * sd, ma - self.bb_mult * sd

    def keltner_channels(self, highs, lows, closes):
        closes = np.asarray(closes)

        if len(closes) < self.ma_period:
            return None, None, None

        ma = np.mean(closes[-self.ma_period:])
        atr = self.atr(highs, lows, closes)

        if atr is None:
            return None, None, None

        return ma, ma + self.kc_mult * atr, ma - self.kc_mult * atr

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

    def is_squeeze(self, highs, lows, closes):
        _, bb_upper, bb_lower = self.bollinger_bands(closes)
        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        if bb_upper is None or kc_upper is None:
            return False

        return bb_upper < kc_upper and bb_lower > kc_lower

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        if not self.is_squeeze(highs, lows, closes):
            return 0

        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        price = np.asarray(closes)[-1]

        if price > kc_upper:
            return 1

        if price < kc_lower:
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

        _, bb_upper, bb_lower = self.bollinger_bands(closes)
        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        return {
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
            "is_squeeze": self.is_squeeze(highs, lows, closes),
        }
