"""
Time-of-Day Pattern System

Concept
-------
Position within the recent N-bar price range as a momentum proxy.
When price is in the upper extreme of the recent range, momentum
is bullish; in the lower extreme, bearish.

Signal logic
------------
range_high = max(highs[-range_period:])
range_low  = min(lows[-range_period:])
pct = (close - range_low) / (range_high - range_low)

If pct > 1 - threshold_pct -> LONG
If pct < threshold_pct     -> SHORT
Otherwise                  -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class TimeOfDayPatternSystem(TradingSystem):

    def __init__(
        self,
        range_period: int = 20,
        threshold_pct: float = 0.20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.range_period = range_period
        self.threshold_pct = threshold_pct
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

    def signal(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        if len(closes) < self.range_period:
            return 0

        range_high = np.max(highs[-self.range_period:])
        range_low = np.min(lows[-self.range_period:])
        span = range_high - range_low

        if span == 0:
            return 0

        pct = (closes[-1] - range_low) / span

        if pct > 1 - self.threshold_pct:
            return 1
        if pct < self.threshold_pct:
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
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])

        if len(closes) < self.range_period:
            return {"range_high": None, "range_low": None, "pct_position": None}

        range_high = float(np.max(highs[-self.range_period:]))
        range_low = float(np.min(lows[-self.range_period:]))
        span = range_high - range_low
        pct = float((closes[-1] - range_low) / span) if span != 0 else None

        return {"range_high": range_high, "range_low": range_low, "pct_position": pct}
