"""
ATR Breakout System

Concept
-------
Detects volatility expansion using the Average True Range (ATR).

Signal logic
------------
Price > Previous Close + k * ATR  → LONG
Price < Previous Close - k * ATR  → SHORT
Otherwise → FLAT

This system captures volatility-driven price expansions.

Position sizing
---------------
ATR-normalized position sizing so that position risk remains
consistent across instruments with different volatility levels.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ATRBreakoutSystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        breakout_mult: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.breakout_mult = breakout_mult
        self.risk_per_trade = risk_per_trade

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
    # Breakout Level
    # ---------------------------------------------------------

    def breakout_levels(self, highs, lows, closes):

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return None, None

        prev_close = closes[-2]

        upper = prev_close + self.breakout_mult * atr
        lower = prev_close - self.breakout_mult * atr

        return upper, lower

    # ---------------------------------------------------------
    # Core Interface
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

        upper, lower = self.breakout_levels(highs, lows, closes)

        if upper is None:
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

        risk_amount = equity * risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, data):

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
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)
        upper, lower = self.breakout_levels(highs, lows, closes)

        return {
            "atr": atr,
            "upper_breakout": upper,
            "lower_breakout": lower,
        }
