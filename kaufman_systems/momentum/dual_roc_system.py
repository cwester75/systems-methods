"""
Dual Rate-of-Change Momentum System

Concept
-------
Uses two momentum measures:

Fast ROC → short-term momentum
Slow ROC → longer-term momentum

Signal logic
------------
ROC_fast > ROC_slow → LONG
ROC_fast < ROC_slow → SHORT

Momentum crossover detects acceleration or deceleration of trend.

Position sizing
---------------
ATR-normalized risk sizing.
"""

import numpy as np


class DualROCSystem:

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be < slow_period")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Rate of Change
    # ---------------------------------------------------------

    def roc(self, prices, period):
        prices = np.asarray(prices)

        if len(prices) < period + 1:
            return None

        prev_price = prices[-1 - period]
        curr_price = prices[-1]

        if prev_price == 0:
            return 0

        return (curr_price - prev_price) / prev_price

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
        1  → Long
       -1  → Short
        0  → Flat
        """

        roc_fast = self.roc(closes, self.fast_period)
        roc_slow = self.roc(closes, self.slow_period)

        if roc_fast is None or roc_slow is None:
            return 0

        if roc_fast > roc_slow:
            return 1

        if roc_fast < roc_slow:
            return -1

        return 0

    def position_sizing(self, equity, highs, lows, closes):
        """
        ATR risk-based sizing
        """

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * self.risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, highs, lows, closes):
        """
        Basic volatility filter
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

    def indicators(self, closes):

        return {
            "roc_fast": self.roc(closes, self.fast_period),
            "roc_slow": self.roc(closes, self.slow_period),
        }
