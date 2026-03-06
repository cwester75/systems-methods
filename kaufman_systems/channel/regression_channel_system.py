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
# channel/regression_channel_system.py

import numpy as np
import pandas as pd


class RegressionChannelSystem:
    """
    Linear Regression Channel Trading System

    Concept
    -------
    Fit a rolling linear regression to price.
    Create upper/lower channel bands using the regression residual standard deviation.

    Upper Channel = regression line + k * std(residuals)
    Lower Channel = regression line - k * std(residuals)

    Trading Logic
    -------------
    Long Entry:
        price crosses above upper channel

    Short Entry:
        price crosses below lower channel

    Exit:
        price crosses back through regression line
    """

    def __init__(
        self,
        lookback: int = 50,
        channel_multiplier: float = 2.0,
        atr_length: int = 14,
        risk_per_trade: float = 0.01,
    ):

        self.lookback = lookback
        self.channel_multiplier = channel_multiplier
        self.atr_length = atr_length
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # Regression Channel
    # -------------------------------------------------

    def regression_channel(self, data: pd.DataFrame):

        prices = data["close"]

        reg_line = pd.Series(index=prices.index, dtype=float)
        upper = pd.Series(index=prices.index, dtype=float)
        lower = pd.Series(index=prices.index, dtype=float)

        for i in range(self.lookback, len(prices)):

            window = prices.iloc[i - self.lookback:i]

            x = np.arange(len(window))
            y = window.values

            slope, intercept = np.polyfit(x, y, 1)

            regression_values = intercept + slope * x
            residuals = y - regression_values

            std = np.std(residuals)

            reg_value = intercept + slope * (len(window) - 1)

            reg_line.iloc[i] = reg_value
            upper.iloc[i] = reg_value + self.channel_multiplier * std
            lower.iloc[i] = reg_value - self.channel_multiplier * std

        return reg_line, upper, lower

    # -------------------------------------------------
    # ATR
    # -------------------------------------------------

    def atr(self, data: pd.DataFrame):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr.rolling(self.atr_length).mean()

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        reg_line, upper, lower = self.regression_channel(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > upper.shift(1)
        short_entry = close < lower.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit conditions
        exit_long = close < reg_line.shift(1)
        exit_short = close > reg_line.shift(1)

        signal[exit_long] = 0
        signal[exit_short] = 0

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / atr

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

        vol_ratio = atr / data["close"]

        # avoid extremely quiet markets
        mask = vol_ratio > 0.004

        return mask

    # -------------------------------------------------
    # Run Backtest
    # -------------------------------------------------

    def run(self, data: pd.DataFrame, capital: float = 100000):

        sig = self.signal(data)
        size = self.position_sizing(data, capital)
        risk_mask = self.risk_filter(data)

        position = sig * size
        position = position.where(risk_mask, 0)

        returns = data["close"].pct_change()

        strat_returns = position.shift(1) * returns

        equity = (1 + strat_returns.fillna(0)).cumprod() * capital

        results = pd.DataFrame(
            {
                "signal": sig,
                "position": position,
                "strategy_returns": strat_returns,
                "equity": equity,
            }
        )

        return results
