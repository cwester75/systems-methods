"""
Price Channel Breakout System

Concept
-------
Classic N-period price channel breakout.  The channel is defined by the
highest high and lowest low over a lookback window, excluding the
current bar.  A breakout occurs when the current close exceeds the
channel boundary.

Signal logic
------------
Close > Highest High (lookback, excluding current bar) → LONG
Close < Lowest Low   (lookback, excluding current bar) → SHORT
Otherwise                                              → FLAT

This differs from the Donchian system by explicitly excluding the
current bar from the channel calculation, giving a cleaner breakout
signal on the bar that actually breaks the level.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PriceChannelBreakoutSystem(TradingSystem):

    def __init__(
        self,
        channel_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
# channel/price_channel_breakout.py

import numpy as np
import pandas as pd


class PriceChannelBreakout:
    """
    Price Channel Breakout System
    ------------------------------
    Classic channel system described in Kaufman-style trend approaches.

    Logic
    -----
    Long Entry:
        price > highest high of last N periods

    Short Entry:
        price < lowest low of last N periods

    Exit:
        Opposite channel break or trailing stop.

    Parameters
    ----------
    channel_length : int
        Lookback window for channel calculation.

    exit_length : int
        Lookback window for exit channel (usually shorter).

    risk_per_trade : float
        Fraction of portfolio risked per trade.

    atr_multiplier : float
        Stop distance in ATR multiples.
    """

    def __init__(
        self,
        channel_length: int = 20,
        exit_length: int = 10,
        risk_per_trade: float = 0.01,
        atr_multiplier: float = 2.0,
    ):

        self.channel_length = channel_length
        self.exit_length = exit_length
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def channel_high(self, highs):
        highs = np.asarray(highs)

        if len(highs) < self.channel_period + 1:
            return None

        return np.max(highs[-(self.channel_period + 1):-1])

    def channel_low(self, lows):
        lows = np.asarray(lows)

        if len(lows) < self.channel_period + 1:
            return None

        return np.min(lows[-(self.channel_period + 1):-1])

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

        ch_high = self.channel_high(highs)
        ch_low = self.channel_low(lows)

        if ch_high is None or ch_low is None:
            return 0

        price = closes[-1]

        if price > ch_high:
            return 1

        if price < ch_low:
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
            "channel_high": self.channel_high(highs),
            "channel_low": self.channel_low(lows),
        }
    def price_channels(self, data: pd.DataFrame):

        high_channel = data["high"].rolling(self.channel_length).max()
        low_channel = data["low"].rolling(self.channel_length).min()

        exit_high = data["high"].rolling(self.exit_length).max()
        exit_low = data["low"].rolling(self.exit_length).min()

        return high_channel, low_channel, exit_high, exit_low

    def atr(self, data: pd.DataFrame, length: int = 14):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(length).mean()

        return atr

    # ---------------------------------------------------------
    # Signal Generation
    # ---------------------------------------------------------

    def signal(self, data: pd.DataFrame):

        high_channel, low_channel, exit_high, exit_low = self.price_channels(data)

        close = data["close"]

        signal = pd.Series(index=data.index, dtype=float)

        long_signal = close > high_channel.shift(1)
        short_signal = close < low_channel.shift(1)

        signal[long_signal] = 1
        signal[short_signal] = -1

        signal = signal.ffill().fillna(0)

        # Exit conditions
        exit_long = close < exit_low.shift(1)
        exit_short = close > exit_high.shift(1)

        signal[exit_long] = 0
        signal[exit_short] = 0

        return signal

    # ---------------------------------------------------------
    # Position Sizing
    # ---------------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)
        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_multiplier)

        return position_size

    # ---------------------------------------------------------
    # Risk Filter
    # ---------------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

        vol_filter = atr / data["close"]

        # avoid extremely low volatility markets
        filter_mask = vol_filter > 0.005

        return filter_mask

    # ---------------------------------------------------------
    # Backtest Helper
    # ---------------------------------------------------------

    def run(self, data: pd.DataFrame, capital: float = 100000):

        sig = self.signal(data)
        size = self.position_sizing(data, capital)
        risk = self.risk_filter(data)

        position = sig * size
        position = position.where(risk, 0)

        returns = data["close"].pct_change()
        strategy_returns = position.shift(1) * returns

        equity = (1 + strategy_returns.fillna(0)).cumprod() * capital

        results = pd.DataFrame(
            {
                "signal": sig,
                "position": position,
                "returns": strategy_returns,
                "equity": equity,
            }
        )

        return results
