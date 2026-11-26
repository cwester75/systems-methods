"""
Opening Gap Pattern System

Concept
-------
Large overnight gaps often lead to continuation moves.  Since we work
with daily closes/highs/lows only, the gap is approximated as the
distance between the current bar's low (gap up) or high (gap down)
and the prior close.

Signal logic
------------
If low[-1] > close[-2] + gap_threshold * ATR  -> LONG  (gap up)
If high[-1] < close[-2] - gap_threshold * ATR -> SHORT (gap down)
Otherwise                                     -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OpeningGapSystem(TradingSystem):

    def __init__(
        self,
        gap_threshold: float = 1.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.gap_threshold = gap_threshold
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

        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return 0

        if len(closes) < 2:
            return 0

        prev_close = closes[-2]
        gap_up = lows[-1] - prev_close
        gap_down = prev_close - highs[-1]

        if gap_up > self.gap_threshold * atr_val:
            return 1
        if gap_down > self.gap_threshold * atr_val:
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
        atr_val = self.atr(highs, lows, closes)
        gap_up = float(lows[-1] - closes[-2]) if len(closes) >= 2 else None
        gap_down = float(closes[-2] - highs[-1]) if len(closes) >= 2 else None
        return {"atr": atr_val, "gap_up": gap_up, "gap_down": gap_down}
