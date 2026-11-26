"""
Fuzzy Logic System

Concept
-------
Replaces hard thresholds with graded membership.  Trend strength
(distance of price from its SMA, in ATR units) is fuzzified into
NEGATIVE / POSITIVE sets, and volatility (short ATR / long ATR) into a
HIGH set.  The rule base is:

  IF trend is POSITIVE THEN bullish
  IF trend is NEGATIVE THEN bearish
  IF volatility is HIGH THEN dampen conviction

Defuzzified score = (mu_pos - mu_neg) * (1 - 0.5 * mu_high_vol)

Signal logic
------------
score > +threshold → LONG
score < -threshold → SHORT
Otherwise          → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class FuzzyLogicSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 30,
        vol_short: int = 5,
        vol_long: int = 20,
        threshold: float = 0.4,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Fuzzy machinery
    # ---------------------------------------------------------

    @staticmethod
    def _ramp(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def _atr_n(self, highs, lows, closes, n):
        highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
        if len(closes) < n + 1:
            return None
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-n:])

    def fuzzy_state(self, data):
        highs, lows = data["highs"], data["lows"]
        closes = np.asarray(data["closes"], dtype=float)

        need = max(self.ma_period, self.vol_long + 1, self.atr_period + 1)
        if len(closes) < need:
            return None

        atr = self._atr_n(highs, lows, closes, self.atr_period)
        a_short = self._atr_n(highs, lows, closes, self.vol_short)
        a_long = self._atr_n(highs, lows, closes, self.vol_long)
        if not atr or not a_long:
            return None

        trend = (closes[-1] - np.mean(closes[-self.ma_period:])) / atr
        vol_ratio = a_short / a_long

        mu_pos = self._ramp(trend, 0.5, 2.0)
        mu_neg = self._ramp(-trend, 0.5, 2.0)
        mu_high_vol = self._ramp(vol_ratio, 1.0, 2.0)
        score = (mu_pos - mu_neg) * (1.0 - 0.5 * mu_high_vol)

        return {
            "trend": float(trend),
            "vol_ratio": float(vol_ratio),
            "mu_positive": mu_pos,
            "mu_negative": mu_neg,
            "mu_high_vol": mu_high_vol,
            "score": float(score),
        }

    def atr(self, highs, lows, closes):
        return self._atr_n(highs, lows, closes, self.atr_period)

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        state = self.fuzzy_state(data)
        if state is None:
            return 0
        if state["score"] > self.threshold:
            return 1
        if state["score"] < -self.threshold:
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
        state = self.fuzzy_state(data)
        if state is None:
            return {"trend": None, "vol_ratio": None, "mu_positive": None,
                    "mu_negative": None, "mu_high_vol": None, "score": None}
        return state
