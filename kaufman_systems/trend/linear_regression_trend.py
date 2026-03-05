"""
Linear Regression Trend System

Concept
-------
Uses the slope of a rolling linear regression line to detect trend direction.
Positive slope → upward trend → long
Negative slope → downward trend → short

Optional confirmation:
R² filter to avoid weak trends.

Position sizing:
ATR-based risk model (consistent with other Kaufman systems).
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class LinearRegressionTrendSystem(TradingSystem):

    def __init__(
        self,
        window: int = 20,
        r2_threshold: float = 0.3,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.window = window
        self.r2_threshold = r2_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Regression indicators
    # ---------------------------------------------------------

    def regression(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.window:
            return None, None

        y = prices[-self.window:]
        x = np.arange(self.window)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Compute R²
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r2 = 0
        else:
            r2 = 1 - ss_res / ss_tot

        return slope, r2

    # ---------------------------------------------------------
    # ATR calculation
    # ---------------------------------------------------------

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
        """
        Returns
        -------
        1  → Long
       -1  → Short
        0  → Flat
        """

        closes = data["closes"]
        slope, r2 = self.regression(closes)

        if slope is None:
            return 0

        if r2 < self.r2_threshold:
            return 0

        if slope > 0:
            return 1
        elif slope < 0:
            return -1
        else:
            return 0

    def position_sizing(self, data, risk):
        """
        ATR-based position sizing
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
        Basic volatility sanity check
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

        closes = data["closes"]
        slope, r2 = self.regression(closes)

        return {
            "regression_slope": slope,
            "regression_r2": r2,
        }
