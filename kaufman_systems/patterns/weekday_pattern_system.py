"""
Weekday Pattern System

Concept
-------
Some assets show consistent return bias by position within a 5-bar
cycle (approximating weekdays).  The system measures rolling average
returns per cycle position and trades when the current position shows
a statistically significant positive or negative bias.

Signal logic
------------
Compute average return per cycle position over the lookback window.
If current position avg return > threshold  -> LONG
If current position avg return < -threshold -> SHORT
Otherwise                                   -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class WeekdayPatternSystem(TradingSystem):

    def __init__(
        self,
        cycle_length: int = 5,
        lookback: int = 100,
        threshold: float = 0.001,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.cycle_length = cycle_length
        self.lookback = lookback
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

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

    def _weekday_avg_return(self, closes):
        closes = np.asarray(closes)
        if len(closes) < self.lookback + 1:
            return None

        returns = np.diff(closes[-self.lookback - 1:]) / closes[-self.lookback - 1:-1]
        n = len(returns)
        current_pos = (n - 1) % self.cycle_length
        positions = np.arange(n) % self.cycle_length
        mask = positions == current_pos
        if not np.any(mask):
            return None
        return float(np.mean(returns[mask]))

    def signal(self, data):
        closes = data["closes"]
        if len(np.asarray(closes)) < self.lookback + 1:
            return 0

        avg_ret = self._weekday_avg_return(closes)
        if avg_ret is None:
            return 0

        if avg_ret > self.threshold:
            return 1
        if avg_ret < -self.threshold:
            return -1
        return 0

    def position_sizing(self, data, risk):
        highs, lows, closes = data["highs"], data["lows"], data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return 0
        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        atr_val = self.atr(data["highs"], data["lows"], data["closes"])
        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        closes = data["closes"]
        avg_ret = self._weekday_avg_return(closes)
        n = len(np.asarray(closes))
        current_pos = (n - 1) % self.cycle_length if n > 0 else None
        return {"cycle_position": current_pos, "avg_return_at_position": avg_ret}
