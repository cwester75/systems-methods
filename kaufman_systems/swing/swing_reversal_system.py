"""
Swing Reversal System

Concept
-------
Identifies swing highs and swing lows using a simple N-bar pivot
definition.  A swing high is a bar whose high is the highest of
2N+1 bars centered on it.  A swing low is the mirror.

Signal logic
------------
When the most recent confirmed swing point is a swing low  → LONG
When the most recent confirmed swing point is a swing high → SHORT

This captures trend reversals at structural turning points.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class SwingReversalSystem(TradingSystem):

    def __init__(
        self,
        swing_period: int = 5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.swing_period = swing_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def last_swing(self, highs, lows):
        """Return ('high', index) or ('low', index) or (None, None)."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        n = self.swing_period

        if len(highs) < 2 * n + 1:
            return None, None

        last_type = None
        last_idx = None

        for i in range(len(highs) - 1, n - 1, -1):
            start = i - n
            end = i + 1

            if start < 0:
                break

            window_highs = highs[max(start, 0):min(end + n, len(highs))]
            window_lows = lows[max(start, 0):min(end + n, len(lows))]

            if i + n < len(highs):
                if highs[i] == np.max(highs[i - n:i + n + 1]):
                    return "high", i

                if lows[i] == np.min(lows[i - n:i + n + 1]):
                    return "low", i

        return None, None

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

        swing_type, swing_idx = self.last_swing(
            np.asarray(highs), np.asarray(lows)
        )

        if swing_type is None:
            return 0

        if swing_type == "low":
            return 1

        if swing_type == "high":
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

        swing_type, swing_idx = self.last_swing(
            np.asarray(highs), np.asarray(lows)
        )

        return {
            "swing_type": swing_type,
            "swing_index": swing_idx,
        }
