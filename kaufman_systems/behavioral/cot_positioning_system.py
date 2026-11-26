"""
COT Positioning System (Price-Derived Proxy)

Concept
-------
Approximates Commitments of Traders positioning via cumulative momentum.
When cumulative ROC reaches an extreme percentile of its own history,
the market is "crowded" and a contrarian signal fires.

Signal logic
------------
cumulative_roc = sum of bar-to-bar ROC over lookback
If cum_roc > upper percentile of its history → crowded long  → SHORT
If cum_roc < lower percentile of its history → crowded short → LONG
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class COTPositioningSystem(TradingSystem):

    def __init__(
        self,
        roc_period: int = 10,
        lookback: int = 50,
        upper_pct: float = 80.0,
        lower_pct: float = 20.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.roc_period = roc_period
        self.lookback = lookback
        self.upper_pct = upper_pct
        self.lower_pct = lower_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _cumulative_momentum(self, closes):
        closes = np.asarray(closes)
        need = self.lookback + self.roc_period
        if len(closes) < need + 1:
            return None, None, None
        roc = (closes[self.roc_period:] - closes[:-self.roc_period]) / closes[:-self.roc_period]
        cum_series = np.array([
            np.sum(roc[max(0, i - self.lookback + 1):i + 1])
            for i in range(len(roc))
        ])
        current = cum_series[-1]
        history = cum_series[-self.lookback:]
        upper = np.percentile(history, self.upper_pct)
        lower = np.percentile(history, self.lower_pct)
        return current, upper, lower

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
        current, upper, lower = self._cumulative_momentum(data["closes"])
        if current is None:
            return 0
        if current > upper:
            return -1
        if current < lower:
            return 1
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
        current, upper, lower = self._cumulative_momentum(data["closes"])
        return {
            "cumulative_momentum": current,
            "upper_threshold": upper,
            "lower_threshold": lower,
        }
