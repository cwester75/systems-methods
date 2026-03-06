"""
VIX-Style Expansion System

Concept
-------
Constructs a VIX-like implied volatility proxy from the rolling
standard deviation of log returns, then signals when this measure
expands sharply.  Sharp expansion from low levels often precedes
directional moves.

Since we don't have options data, we use realized volatility
(annualized std of log returns) as a proxy.

Signal logic
------------
If current realized vol > expansion_mult × SMA(realized vol, lookback):
  Close > prev close → LONG
  Close < prev close → SHORT
Otherwise            → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class VIXExpansionSystem(TradingSystem):

    def __init__(
        self,
        vol_window: int = 20,
        lookback: int = 50,
        expansion_mult: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.vol_window = vol_window
        self.lookback = lookback
        self.expansion_mult = expansion_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def realized_vol(self, closes):
        closes = np.asarray(closes, dtype=float)

        if len(closes) < self.vol_window + 1:
            return None

        log_returns = np.log(closes[1:] / closes[:-1])

        return np.std(log_returns[-self.vol_window:])

    def vol_series(self, closes):
        """Rolling realized vol for last `lookback` periods."""
        closes = np.asarray(closes, dtype=float)

        needed = self.vol_window + 1 + self.lookback

        if len(closes) < needed:
            return None

        log_returns = np.log(closes[1:] / closes[:-1])

        vols = []
        for i in range(self.lookback):
            end = len(log_returns) - i
            start = end - self.vol_window
            vols.append(np.std(log_returns[start:end]))

        return np.array(vols[::-1])

    def atr(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

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
        closes = data["closes"]
        closes = np.asarray(closes)

        vol_vals = self.vol_series(closes)

        if vol_vals is None:
            return 0

        current_vol = vol_vals[-1]
        avg_vol = np.mean(vol_vals)

        if avg_vol == 0:
            return 0

        if current_vol <= self.expansion_mult * avg_vol:
            return 0

        if len(closes) < 2:
            return 0

        if closes[-1] > closes[-2]:
            return 1

        if closes[-1] < closes[-2]:
            return -1

        return 0

    def position_sizing(self, data, risk):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None or atr <= 0:
            return False

        return True

    def indicators(self, data):
        closes = data["closes"]

        vol_vals = self.vol_series(closes)

        if vol_vals is None:
            return {"current_vol": None, "avg_vol": None, "is_expansion": False}

        current_vol = vol_vals[-1]
        avg_vol = np.mean(vol_vals)

        return {
            "current_vol": current_vol,
            "avg_vol": avg_vol,
            "is_expansion": avg_vol > 0 and current_vol > self.expansion_mult * avg_vol,
        }
