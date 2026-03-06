"""
RSI Reversal System

Concept
-------
Mean-reversion strategy using the Relative Strength Index (RSI).

Signal logic
------------
RSI < oversold_threshold  → LONG
RSI > overbought_threshold → SHORT
Otherwise → FLAT

Exit logic (handled by signal reversal or neutral RSI zone).

Position sizing
---------------
ATR-based risk normalization to standardize position exposure.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RSIReversalSystem(TradingSystem):

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # RSI calculation
    # ---------------------------------------------------------

    def rsi(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.rsi_period + 1:
            return None

        deltas = np.diff(prices[-(self.rsi_period + 1):])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # ---------------------------------------------------------
    # ATR calculation
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
    # Core system interface
    # ---------------------------------------------------------

    def signal(self, data):
        """
        Returns
        -------
        1  → Long
       -1  → Short
        0  → Flat
        """

        closes = data["closes"]
        rsi_val = self.rsi(closes)

        if rsi_val is None:
            return 0

        if rsi_val < self.oversold:
            return 1

        if rsi_val > self.overbought:
            return -1

        return 0

    def position_sizing(self, data, risk):
        """
        ATR-based position sizing
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
        Basic volatility sanity check
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

        closes = data["closes"]

        return {
            "rsi": self.rsi(closes)
        }
