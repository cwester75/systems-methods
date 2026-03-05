"""
Efficiency Ratio Trend System
Based on concepts from Perry Kaufman.

Signal logic
------------
ER = |Price_t - Price_t-n| / sum(|ΔPrice|)

High ER → persistent directional movement → trend-following
Low ER → noisy movement → no trade

System behavior
---------------
ER > threshold and price rising → LONG
ER > threshold and price falling → SHORT
Otherwise → FLAT
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ERTrendSystem(TradingSystem):
    def __init__(
        self,
        er_period: int = 10,
        er_threshold: float = 0.4,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.er_period = er_period
        self.er_threshold = er_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicator calculations
    # ---------------------------------------------------------

    def efficiency_ratio(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.er_period + 1:
            return None

        change = abs(prices[-1] - prices[-1 - self.er_period])
        volatility = np.sum(np.abs(np.diff(prices[-1 - self.er_period :])))

        if volatility == 0:
            return 0

        return change / volatility

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
    # Core system interface
    # ---------------------------------------------------------

    def signal(self, data):
        """
        Returns:
            1  -> Long
           -1  -> Short
            0  -> Flat
        """

        closes = data["closes"]
        er = self.efficiency_ratio(closes)

        if er is None:
            return 0

        if er < self.er_threshold:
            return 0

        momentum = closes[-1] - closes[-self.er_period]

        if momentum > 0:
            return 1
        elif momentum < 0:
            return -1
        else:
            return 0

    def position_sizing(self, data, risk):
        """
        ATR risk model sizing
        """

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * risk_per_trade
        position = risk_amount / atr

        return position

    def risk_filter(self, data):
        """
        Basic volatility sanity filter
        """

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, data):
        """
        Returns diagnostic values for research / plotting
        """

        closes = data["closes"]

        return {
            "efficiency_ratio": self.efficiency_ratio(closes),
        }
