"""
Regression Channel System

Concept
-------
A channel formed by a linear regression line ± k standard deviations
of the residuals.  Breakouts beyond the channel suggest a shift in the
underlying trend regime.

Signal logic
------------
Close > Regression line + k × StdErr → LONG
Close < Regression line - k × StdErr → SHORT
Otherwise                            → FLAT

The regression channel adapts its slope to the current trend while
the residual bands capture the normal scatter around the trend.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RegressionChannelSystem(TradingSystem):

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.window = window
        self.num_std = num_std
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def regression_channel(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.window:
            return None, None, None

        y = prices[-self.window:]
        x = np.arange(self.window)

        slope, intercept = np.polyfit(x, y, 1)

        y_hat = slope * x + intercept
        residuals = y - y_hat
        std_err = np.std(residuals)

        reg_value = y_hat[-1]
        upper = reg_value + self.num_std * std_err
        lower = reg_value - self.num_std * std_err

        return reg_value, upper, lower

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

        reg_value, upper, lower = self.regression_channel(closes)

        if reg_value is None:
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

        reg_value, upper, lower = self.regression_channel(closes)

        return {
            "regression_value": reg_value,
            "upper_channel": upper,
            "lower_channel": lower,
        }
