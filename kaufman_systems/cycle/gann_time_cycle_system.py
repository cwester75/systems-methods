"""
Gann Time Cycle System

Concept
-------
Fixed cycle intervals as potential turning points (inspired by Gann).
Every cycle_period bars is a potential turn. Near a turn point, the
signal is determined by price relative to its SMA.

Signal logic
------------
bar_index = len(closes) - 1
If bar_index mod cycle_period is within turn_window:
  close > SMA → LONG
  close < SMA → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class GannTimeCycleSystem(TradingSystem):

    def __init__(
        self,
        cycle_period: int = 30,
        turn_window: int = 2,
        ma_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.cycle_period = cycle_period
        self.turn_window = turn_window
        self.ma_period = ma_period
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
        min_needed = max(self.ma_period, self.atr_period + 1, self.cycle_period)
        if len(closes) < min_needed:
            return 0
        bar_index = len(closes) - 1
        dist_to_cycle = bar_index % self.cycle_period
        near_turn = dist_to_cycle <= self.turn_window or (self.cycle_period - dist_to_cycle) <= self.turn_window
        if not near_turn:
            return 0
        ma = np.mean(closes[-self.ma_period:])
        price = closes[-1]
        if price > ma:
            return 1
        if price < ma:
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
        closes = np.asarray(data["closes"])
        bar_index = len(closes) - 1
        dist_to_cycle = bar_index % self.cycle_period
        near_turn = dist_to_cycle <= self.turn_window or (self.cycle_period - dist_to_cycle) <= self.turn_window
        ma = None
        if len(closes) >= self.ma_period:
            ma = float(np.mean(closes[-self.ma_period:]))
        return {
            "bar_index": bar_index,
            "distance_to_cycle": dist_to_cycle,
            "near_turn": near_turn,
            "ma": ma,
        }
