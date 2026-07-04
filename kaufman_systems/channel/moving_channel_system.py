"""
Moving Channel System

Concept
-------
A channel defined by offset bands around a simple moving average.
The band width is a fixed percentage of the MA value.  Breakouts
above/below the channel trigger trend-following signals.

Signal logic
------------
Close > MA + (MA × band_pct) → LONG
Close < MA - (MA × band_pct) → SHORT
Otherwise                    → FLAT

Kaufman discusses moving-average channels as a way to capture
breakouts while filtering noise within the channel width.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class MovingChannelSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        band_pct: float = 0.03,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.band_pct = band_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def sma(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.ma_period:
            return None

        return np.mean(prices[-self.ma_period:])

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

        ma = self.sma(closes)

        if ma is None:
            return 0

        upper = ma * (1 + self.band_pct)
        lower = ma * (1 - self.band_pct)

        price = closes[-1]

        if price > upper:
            return 1

        if price < lower:
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

        ma = self.sma(closes)

        if ma is None:
            return {"ma": None, "upper_band": None, "lower_band": None}

        return {
            "ma": ma,
            "upper_band": ma * (1 + self.band_pct),
            "lower_band": ma * (1 - self.band_pct),
        }
