"""
Market Sentiment Index System

Concept
-------
Multi-indicator composite that combines momentum, mean reversion, and
volatility state into a single sentiment index. Each component scores
+1, 0, or -1. The composite sum determines the signal.

Signal logic
------------
Momentum score: ROC > 0 → +1, ROC < 0 → -1, else 0
Mean reversion score: z-score > 1 → -1, z-score < -1 → +1, else 0
Volatility score: short ATR > long ATR → +1, else -1
Composite = sum of scores
If composite >= 2 → LONG
If composite <= -2 → SHORT
Otherwise → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class MarketSentimentIndexSystem(TradingSystem):

    def __init__(
        self,
        roc_period: int = 10,
        zscore_period: int = 30,
        vol_short: int = 5,
        vol_long: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.roc_period = roc_period
        self.zscore_period = zscore_period
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def _scores(self, data):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])
        min_needed = max(self.roc_period + 1, self.zscore_period, self.vol_long + 1, self.atr_period + 1)
        if len(closes) < min_needed:
            return None, None, None

        roc = (closes[-1] - closes[-(self.roc_period + 1)]) / closes[-(self.roc_period + 1)]
        mom_score = 1 if roc > 0 else (-1 if roc < 0 else 0)

        window = closes[-self.zscore_period:]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore = (closes[-1] - mean) / std
        else:
            zscore = 0.0
        mr_score = -1 if zscore > 1 else (1 if zscore < -1 else 0)

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        short_atr = np.mean(tr[-self.vol_short:])
        long_atr = np.mean(tr[-self.vol_long:])
        vol_score = 1 if short_atr > long_atr else -1

        return mom_score, mr_score, vol_score

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
        scores = self._scores(data)
        if scores[0] is None:
            return 0
        composite = sum(scores)
        if composite >= 2:
            return 1
        if composite <= -2:
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
        scores = self._scores(data)
        if scores[0] is None:
            return {"momentum_score": None, "mean_reversion_score": None, "volatility_score": None, "composite": None}
        return {
            "momentum_score": scores[0],
            "mean_reversion_score": scores[1],
            "volatility_score": scores[2],
            "composite": sum(scores),
        }
