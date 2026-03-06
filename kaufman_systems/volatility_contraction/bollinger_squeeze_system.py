"""
Bollinger Squeeze System

Concept
-------
Detects Bollinger Band contraction (squeeze) as a precursor to
volatility expansion.  When the bandwidth (distance between upper and
lower bands relative to the MA) falls below a threshold, the market is
in a squeeze.  The breakout direction after the squeeze fires the signal.

Signal logic
------------
If bandwidth < squeeze_threshold:
  Close > Upper Band → LONG
  Close < Lower Band → SHORT
Otherwise            → FLAT

Kaufman discusses Bollinger squeezes as low-risk entry setups where
compressed volatility precedes directional moves.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class BollingerSqueezeSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        band_mult: float = 2.0,
        squeeze_threshold: float = 0.04,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.band_mult = band_mult
        self.squeeze_threshold = squeeze_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def bollinger(self, prices):
        """Compute bands from bars *prior* to the current bar.

        The squeeze is a condition on the recent past; the current bar
        is the one that breaks out.  Using bars[:-1] prevents the
        breakout bar from inflating the bandwidth calculation.
        """
        prices = np.asarray(prices)

        if len(prices) < self.ma_period + 1:
            return None, None, None, None

        window = prices[-(self.ma_period + 1):-1]
        ma = np.mean(window)
        sd = np.std(window)

        upper = ma + self.band_mult * sd
        lower = ma - self.band_mult * sd
        bandwidth = (upper - lower) / ma if ma != 0 else None

        return ma, upper, lower, bandwidth

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

        ma, upper, lower, bandwidth = self.bollinger(closes)

        if bandwidth is None:
            return 0

        if bandwidth >= self.squeeze_threshold:
            return 0

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

        ma, upper, lower, bandwidth = self.bollinger(closes)

        return {
            "ma": ma,
            "upper_band": upper,
            "lower_band": lower,
            "bandwidth": bandwidth,
            "is_squeeze": bandwidth is not None and bandwidth < self.squeeze_threshold,
        }
