"""
Fibonacci Projection System

Concept
-------
Identifies Fibonacci retracement levels from recent swing high and low.
Price above the 61.8% level suggests strong uptrend continuation;
price below the 38.2% level suggests downtrend continuation.

Signal logic
------------
swing_high = max(closes over swing_period)
swing_low  = min(closes over swing_period)
fib_382 = swing_low + 0.382 × (swing_high - swing_low)
fib_618 = swing_low + 0.618 × (swing_high - swing_low)

If close > fib_618 → LONG
If close < fib_382 → SHORT
Otherwise          → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class FibonacciProjectionSystem(TradingSystem):

    def __init__(
        self,
        swing_period: int = 50,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.swing_period = swing_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _fib_levels(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.swing_period + 1:
            return None, None, None, None
        window = closes[-(self.swing_period + 1):-1]
        swing_high = np.max(window)
        swing_low = np.min(window)
        rng = swing_high - swing_low
        if rng == 0:
            return swing_high, swing_low, None, None
        fib_382 = swing_low + 0.382 * rng
        fib_618 = swing_low + 0.618 * rng
        return swing_high, swing_low, fib_382, fib_618

    def atr(self, highs, lows, closes):
        highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
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

    def signal(self, data):
        closes = np.asarray(data["closes"])
        swing_high, swing_low, fib_382, fib_618 = self._fib_levels(closes)
        if fib_382 is None:
            return 0
        price = closes[-1]
        if price > fib_618:
            return 1
        if price < fib_382:
            return -1
        return 0

    def position_sizing(self, data, risk):
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val == 0:
            return 0
        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        swing_high, swing_low, fib_382, fib_618 = self._fib_levels(data["closes"])
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "fib_382": fib_382,
            "fib_618": fib_618,
        }
