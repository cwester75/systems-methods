"""
High-Low Channel System

Concept
-------
Channel defined by separate moving averages of the highs and lows.
The upper band is the SMA of highs; the lower band is the SMA of lows.
This creates a natural volatility-adaptive envelope around price.

Signal logic
------------
Close > SMA(Highs, N) → LONG
Close < SMA(Lows, N)  → SHORT
Otherwise             → FLAT

Kaufman describes high-low channels as a structural way to define
support and resistance without arbitrary band offsets.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class HighLowChannelSystem(TradingSystem):

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def high_channel(self, highs):
        highs = np.asarray(highs)

        if len(highs) < self.channel_period:
            return None

        return np.mean(highs[-self.channel_period:])

    def low_channel(self, lows):
        lows = np.asarray(lows)

        if len(lows) < self.channel_period:
            return None

        return np.mean(lows[-self.channel_period:])

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
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        upper = self.high_channel(highs)
        lower = self.low_channel(lows)

        if upper is None or lower is None:
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
        highs = data["highs"]
        lows = data["lows"]

        return {
            "high_channel": self.high_channel(highs),
            "low_channel": self.low_channel(lows),
        }
# channel/high_low_channel_system.py

import numpy as np
import pandas as pd


class HighLowChannelSystem:
    """
    High–Low Channel Trading System

    Concept
    -------
    Defines a price channel using the rolling highest high
    and lowest low.

    Upper Channel = highest(high, N)
    Lower Channel = lowest(low, N)

    Trading Logic
    -------------
    Long Entry:
        close > previous upper channel

    Short Entry:
        close < previous lower channel

    Exit:
        price crosses the mid-channel or opposite breakout
    """

    def __init__(
        self,
        channel_length: int = 20,
        exit_length: int = 10,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.channel_length = channel_length
        self.exit_length = exit_length
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -----------------------------------------------------
    # Channel Calculation
    # -----------------------------------------------------

    def channels(self, data: pd.DataFrame):

        upper = data["high"].rolling(self.channel_length).max()
        lower = data["low"].rolling(self.channel_length).min()

        exit_upper = data["high"].rolling(self.exit_length).max()
        exit_lower = data["low"].rolling(self.exit_length).min()

        mid = (upper + lower) / 2

        return upper, lower, mid, exit_upper, exit_lower

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    def atr(self, data: pd.DataFrame):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr.rolling(self.atr_length).mean()

    # -----------------------------------------------------
    # Signal Logic
    # -----------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        upper, lower, mid, exit_upper, exit_lower = self.channels(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > upper.shift(1)
        short_entry = close < lower.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit rules
        exit_long = close < exit_lower.shift(1)
        exit_short = close > exit_upper.shift(1)

        signal[exit_long] = 0
        signal[exit_short] = 0

        return signal

    # -----------------------------------------------------
    # Position Sizing
    # -----------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_multiplier)

        return position_size

    # -----------------------------------------------------
    # Risk Filter
    # -----------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

        vol_ratio = atr / data["close"]

        # avoid extremely quiet markets
        mask = vol_ratio > 0.004

        return mask

    # -----------------------------------------------------
    # Run Backtest
    # -----------------------------------------------------

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
