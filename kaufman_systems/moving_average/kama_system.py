"""
Kaufman Adaptive Moving Average (KAMA) System

Concept
-------
KAMA adjusts its smoothing constant using the Efficiency Ratio (ER).

High ER → strong trend → faster response
Low ER → noisy market → slower response

Signal logic
------------
Price > KAMA → LONG
Price < KAMA → SHORT

Position sizing
---------------
ATR-based risk normalization.
"""

import numpy as np


class KAMASystem:

    def __init__(
        self,
        er_period: int = 10,
        fast_period: int = 2,
        slow_period: int = 30,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.er_period = er_period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Efficiency Ratio
    # ---------------------------------------------------------

    def efficiency_ratio(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.er_period + 1:
            return None

        change = abs(prices[-1] - prices[-1 - self.er_period])
        volatility = np.sum(np.abs(np.diff(prices[-1 - self.er_period:])))

        if volatility == 0:
            return 0

        return change / volatility

    # ---------------------------------------------------------
    # KAMA Calculation
    # ---------------------------------------------------------

    def kama(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.er_period + 1:
            return None

        er = self.efficiency_ratio(prices)

        if er is None:
            return None

        fast_sc = 2 / (self.fast_period + 1)
        slow_sc = 2 / (self.slow_period + 1)

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama_val = prices[-self.er_period]

        for price in prices[-self.er_period + 1:]:
            kama_val = kama_val + sc * (price - kama_val)

        return kama_val

    # ---------------------------------------------------------
    # ATR
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
    # Core Interface
    # ---------------------------------------------------------

    def signal(self, closes):

        kama_val = self.kama(closes)

        if kama_val is None:
            return 0

        price = closes[-1]

        if price > kama_val:
            return 1

        if price < kama_val:
            return -1

        return 0

    def position_sizing(self, equity, highs, lows, closes):

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * self.risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, highs, lows, closes):

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, closes):

        return {
            "kama": self.kama(closes),
            "efficiency_ratio": self.efficiency_ratio(closes),
        }
