# channel/moving_channel_system.py

import numpy as np
import pandas as pd


class MovingChannelSystem:
    """
    Moving Channel Trading System
    ------------------------------
    Uses moving averages of highs and lows to create a dynamic channel.

    Concept
    -------
    Upper Channel  = Moving average of highs
    Lower Channel  = Moving average of lows

    Long Entry:
        close crosses above upper channel

    Short Entry:
        close crosses below lower channel

    Exit:
        cross back inside channel or opposite signal
    """

    def __init__(
        self,
        channel_length: int = 20,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):

        self.channel_length = channel_length
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -----------------------------------------------------
    # Channel Calculation
    # -----------------------------------------------------

    def moving_channels(self, data: pd.DataFrame):

        upper = data["high"].rolling(self.channel_length).mean()
        lower = data["low"].rolling(self.channel_length).mean()
        mid = (upper + lower) / 2

        return upper, lower, mid

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

        upper, lower, mid = self.moving_channels(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > upper.shift(1)
        short_entry = close < lower.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # exit when price crosses mid channel
        exit_long = close < mid.shift(1)
        exit_short = close > mid.shift(1)

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

        # avoid extremely low volatility environments
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
