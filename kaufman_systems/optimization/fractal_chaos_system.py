"""
Fractal / Chaos System

Concept
-------
Estimates the Hurst exponent H with rescaled-range (R/S) analysis to
classify the market regime, then trades accordingly.

  H > trend_h  : persistent (trending)       → follow momentum
  H < revert_h : anti-persistent (reverting) → fade deviation from SMA
  otherwise    : random-walk-like            → FLAT

Signal logic
------------
Trending : LONG if close > close[-dir_period], SHORT if below.
Reverting: SHORT if close > SMA(dir_period), LONG if below.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class FractalChaosSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 64,
        trend_h: float = 0.55,
        revert_h: float = 0.45,
        dir_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.trend_h = trend_h
        self.revert_h = revert_h
        self.dir_period = dir_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    @property
    def min_bars(self):
        return max(self.lookback + 1, self.dir_period + 1, self.atr_period + 1)

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def hurst(self, closes):
        """R/S Hurst exponent of the recent log returns, or None."""
        closes = np.asarray(closes, dtype=float)
        if len(closes) < self.lookback + 1 or np.any(closes <= 0):
            return None

        rets = np.diff(np.log(closes[-self.lookback - 1:]))
        sizes = [s for s in (8, 16, 32) if s <= len(rets)]
        if len(sizes) < 2:
            return None

        log_n, log_rs = [], []
        for size in sizes:
            rs_vals = []
            for start in range(0, len(rets) - size + 1, size):
                chunk = rets[start:start + size]
                s = np.std(chunk)
                if s == 0:
                    continue
                dev = np.cumsum(chunk - np.mean(chunk))
                rs_vals.append((dev.max() - dev.min()) / s)
            if rs_vals:
                log_n.append(np.log(size))
                log_rs.append(np.log(np.mean(rs_vals)))

        if len(log_n) < 2:
            return None
        return float(np.polyfit(log_n, log_rs, 1)[0])

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

    def regime(self, closes):
        h = self.hurst(closes)
        if h is None:
            return None, "unknown"
        if h > self.trend_h:
            return h, "trending"
        if h < self.revert_h:
            return h, "reverting"
        return h, "random"

    def signal(self, data):
        closes = np.asarray(data["closes"], dtype=float)
        if len(closes) < self.min_bars:
            return 0

        _, regime = self.regime(closes)

        if regime == "trending":
            return int(np.sign(closes[-1] - closes[-self.dir_period - 1]))
        if regime == "reverting":
            return int(-np.sign(closes[-1] - np.mean(closes[-self.dir_period:])))
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
        h, regime = self.regime(data["closes"])
        return {"hurst": h, "regime": regime}
