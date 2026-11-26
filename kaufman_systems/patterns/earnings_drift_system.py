"""
Earnings Drift Pattern System

Concept
-------
Large single-bar moves (earnings event proxy) often lead to
continuation drift.  When bar[-2] had a return exceeding
drift_threshold * ATR, the system follows the direction on bar[-1].

Signal logic
------------
event_return = abs(close[-2] - close[-3]) / close[-3]
atr_return   = ATR / close[-3]

If abs(close[-2] - close[-3]) > drift_threshold * ATR:
    close[-2] > close[-3] -> LONG  (upward drift)
    close[-2] < close[-3] -> SHORT (downward drift)
Otherwise                 -> FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class EarningsDriftSystem(TradingSystem):

    def __init__(
        self,
        drift_threshold: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.drift_threshold = drift_threshold
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

        if len(closes) < 3:
            return 0

        atr_val = self.atr(highs, lows, closes)
        if atr_val is None or atr_val == 0:
            return 0

        event_move = closes[-2] - closes[-3]

        if abs(event_move) > self.drift_threshold * atr_val:
            if event_move > 0:
                return 1
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
        event_move = float(closes[-2] - closes[-3]) if len(closes) >= 3 else None
        is_event = (event_move is not None and atr_val is not None and
                    atr_val != 0 and abs(event_move) > self.drift_threshold * atr_val)
        return {"atr": atr_val, "event_move": event_move, "is_event": is_event}
