"""
Expert System

Concept
-------
A rule-based voting system.  Four independent rules each vote
+1 / 0 / -1; a trade fires only when enough rules agree.

Rules
-----
1. Trend      : close vs SMA(ma_period)
2. Momentum   : ROC over roc_period
3. Channel    : close vs midpoint of the prior channel_period high/low range
4. Slope      : SMA(ma_period) now vs slope_lag bars ago

Signal logic
------------
vote total >= +min_votes → LONG
vote total <= -min_votes → SHORT
Otherwise                → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ExpertSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 30,
        roc_period: int = 10,
        channel_period: int = 20,
        slope_lag: int = 5,
        min_votes: int = 3,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.roc_period = roc_period
        self.channel_period = channel_period
        self.slope_lag = slope_lag
        self.min_votes = min_votes
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    @property
    def min_bars(self):
        return max(self.ma_period + self.slope_lag, self.roc_period + 1,
                   self.channel_period + 1, self.atr_period + 1)

    # ---------------------------------------------------------
    # Rule base
    # ---------------------------------------------------------

    def votes(self, data):
        closes = np.asarray(data["closes"], dtype=float)
        highs = np.asarray(data["highs"], dtype=float)
        lows = np.asarray(data["lows"], dtype=float)

        if len(closes) < self.min_bars:
            return None

        sma_now = np.mean(closes[-self.ma_period:])
        sma_prev = np.mean(closes[-self.ma_period - self.slope_lag:-self.slope_lag])
        mid = (np.max(highs[-self.channel_period - 1:-1])
               + np.min(lows[-self.channel_period - 1:-1])) / 2

        return {
            "trend": int(np.sign(closes[-1] - sma_now)),
            "momentum": int(np.sign(closes[-1] - closes[-self.roc_period - 1])),
            "channel": int(np.sign(closes[-1] - mid)),
            "slope": int(np.sign(sma_now - sma_prev)),
        }

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

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        votes = self.votes(data)
        if votes is None:
            return 0
        total = sum(votes.values())
        if total >= self.min_votes:
            return 1
        if total <= -self.min_votes:
            return -1
        return 0

    def position_sizing(self, data, risk):
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr = self.atr(data["highs"], data["lows"], data["closes"])
        if atr is None or atr == 0:
            return 0
        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        atr = self.atr(data["highs"], data["lows"], data["closes"])
        return bool(atr is not None and atr > 0)

    def indicators(self, data):
        votes = self.votes(data)
        if votes is None:
            return {"votes": None, "vote_total": None}
        return {"votes": votes, "vote_total": sum(votes.values())}
