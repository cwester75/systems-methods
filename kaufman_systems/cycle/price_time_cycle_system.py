"""
Price-Time Cycle System

Concept
-------
Estimates the dominant price cycle via autocorrelation of returns.
Finds the lag with the highest positive autocorrelation within
[min_cycle, max_cycle]. If the return at that lag ago was positive,
expect a positive cycle phase now → LONG; negative → SHORT.

Signal logic
------------
returns = diff(closes) / closes[:-1]
For each lag in [min_cycle, max_cycle]:
  autocorrelation(returns, lag)
dominant_lag = argmax of positive autocorrelations
If return[dominant_lag ago] > 0 → LONG
If return[dominant_lag ago] < 0 → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PriceTimeCycleSystem(TradingSystem):

    def __init__(
        self,
        min_cycle: int = 10,
        max_cycle: int = 50,
        lookback: int = 100,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.lookback = lookback
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _dominant_cycle(self, closes):
        closes = np.asarray(closes)
        need = self.lookback + 1
        if len(closes) < need:
            return None
        window = closes[-need:]
        rets = np.diff(window) / window[:-1]
        rets = rets - np.mean(rets)
        var = np.var(rets)
        if var == 0:
            return None
        best_lag = None
        best_corr = 0.0
        for lag in range(self.min_cycle, min(self.max_cycle + 1, len(rets))):
            corr = np.mean(rets[lag:] * rets[:-lag]) / var
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        return best_lag

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
        dominant = self._dominant_cycle(closes)
        if dominant is None:
            return 0
        if len(closes) < dominant + 1:
            return 0
        past_return = closes[-1] - closes[-(dominant + 1)]
        if past_return > 0:
            return 1
        if past_return < 0:
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
        dominant = self._dominant_cycle(data["closes"])
        return {
            "dominant_cycle": dominant,
        }
