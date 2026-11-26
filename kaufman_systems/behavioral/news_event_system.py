"""
News Event System (Price-Derived Proxy)

Concept
-------
Detects large single-bar moves as a proxy for news/event-driven action.
When the absolute price change exceeds a multiple of ATR, the bar is
treated as an event bar and the signal follows the move direction.

Signal logic
------------
move = close[-1] - close[-2]
If |move| > event_threshold × ATR:
  move > 0 → LONG
  move < 0 → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class NewsEventSystem(TradingSystem):

    def __init__(
        self,
        event_threshold: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.event_threshold = event_threshold
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
        if len(closes) < self.atr_period + 2:
            return 0
        atr_val = self.atr(data["highs"], data["lows"], closes)
        if atr_val is None or atr_val == 0:
            return 0
        move = closes[-1] - closes[-2]
        if abs(move) > self.event_threshold * atr_val:
            return 1 if move > 0 else -1
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
        closes = np.asarray(data["closes"])
        atr_val = self.atr(data["highs"], data["lows"], closes)
        move = None
        is_event = False
        if len(closes) >= 2 and atr_val is not None and atr_val > 0:
            move = closes[-1] - closes[-2]
            is_event = abs(move) > self.event_threshold * atr_val
        return {
            "move": move,
            "atr": atr_val,
            "is_event": is_event,
        }
