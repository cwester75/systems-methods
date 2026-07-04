"""
ATR Contraction System

Concept
-------
Detects when ATR contracts to a low relative to its recent history,
signaling volatility compression.  When current ATR drops below a
fraction of its rolling average, a contraction is detected.  The
breakout direction from the prior close determines the signal.

Signal logic
------------
If ATR < contraction_pct × SMA(ATR, lookback):
  Close > prev close + current ATR → LONG
  Close < prev close - current ATR → SHORT
Otherwise                          → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ATRContractionSystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 50,
        contraction_pct: float = 0.75,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.lookback = lookback
        self.contraction_pct = contraction_pct
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
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

    def atr_series(self, highs, lows, closes):
        """Compute rolling ATR for the last `lookback` bars."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        needed = self.atr_period + 1 + self.lookback

        if len(closes) < needed:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        atrs = []
        for i in range(self.lookback):
            end = len(tr) - i
            start = end - self.atr_period
            atrs.append(np.mean(tr[start:end]))

        return np.array(atrs[::-1])

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        closes = np.asarray(closes)

        atr_vals = self.atr_series(highs, lows, closes)

        if atr_vals is None:
            return 0

        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals)

        if avg_atr == 0:
            return 0

        if current_atr >= self.contraction_pct * avg_atr:
            return 0

        if len(closes) < 2:
            return 0

        prev_close = closes[-2]
        price = closes[-1]

        if price > prev_close + current_atr:
            return 1

        if price < prev_close - current_atr:
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

        atr_vals = self.atr_series(highs, lows, closes)

        if atr_vals is None:
            return {"current_atr": None, "avg_atr": None, "is_contraction": False}

        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals)

        return {
            "current_atr": current_atr,
            "avg_atr": avg_atr,
            "is_contraction": current_atr < self.contraction_pct * avg_atr,
        }
