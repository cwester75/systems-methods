"""
Bollinger Band Breakout System

Concept
-------
Uses Bollinger Bands to detect volatility expansion and directional
breakouts.

Signal logic
------------
Price > Upper Band → LONG breakout
Price < Lower Band → SHORT breakout
Otherwise → FLAT

Bands
-----
Upper  = MA + k * StdDev
Lower  = MA - k * StdDev

Position sizing
---------------
ATR-based sizing so that position risk is normalized across assets.
"""

import numpy as np


class BollingerBreakoutSystem:

    def __init__(
        self,
        ma_period: int = 20,
        band_mult: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.band_mult = band_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------------------

    def moving_average(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.ma_period:
            return None

        return np.mean(prices[-self.ma_period:])

    def std_dev(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.ma_period:
            return None

        return np.std(prices[-self.ma_period:])

    def bands(self, prices):

        ma = self.moving_average(prices)
        sd = self.std_dev(prices)

        if ma is None or sd is None:
            return None, None

        upper = ma + self.band_mult * sd
        lower = ma - self.band_mult * sd

        return upper, lower

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
    # Core System Interface
    # ---------------------------------------------------------

    def signal(self, closes):
        """
        Returns
        -------
        1  → Long breakout
       -1  → Short breakout
        0  → No signal
        """

        upper, lower = self.bands(closes)

        if upper is None:
            return 0

        price = closes[-1]

        if price > upper:
            return 1

        if price < lower:
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

        ma = self.moving_average(closes)
        sd = self.std_dev(closes)
        upper, lower = self.bands(closes)

        return {
            "bollinger_ma": ma,
            "bollinger_std": sd,
            "upper_band": upper,
            "lower_band": lower,
        }
