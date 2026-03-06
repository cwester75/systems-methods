# volatility_contraction/standard_deviation_breakout.py

import numpy as np
import pandas as pd


class StandardDeviationBreakout:
    """
    Standard Deviation Breakout System

    Concept
    -------
    Detect price moves that exceed a multiple of the recent
    standard deviation of returns.

    A breakout occurs when price deviates significantly from
    its recent mean.

    Indicators
    ----------
    mean = rolling mean(close)
    std  = rolling std(close)

    upper_band = mean + k * std
    lower_band = mean - k * std

    Trading Logic
    -------------
    Long:
        close > upper_band

    Short:
        close < lower_band

    Exit:
        price crosses back through mean
    """

    def __init__(
        self,
        lookback: int = 20,
        std_multiplier: float = 2.0,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.std_multiplier = std_multiplier
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # Standard Deviation Bands
    # -------------------------------------------------

    def std_bands(self, data: pd.DataFrame):

        close = data["close"]

        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()

        upper = mean + self.std_multiplier * std
        lower = mean - self.std_multiplier * std

        return mean, upper, lower

    # -------------------------------------------------
    # ATR
    # -------------------------------------------------

    def atr(self, data: pd.DataFrame):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_length).mean()

        return atr

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        mean, upper, lower = self.std_bands(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > upper.shift(1)
        short_entry = close < lower.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit conditions
        exit_long = close < mean.shift(1)
        exit_short = close > mean.shift(1)

        signal[exit_long] = 0
        signal[exit_short] = 0

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_multiplier)

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

        vol_ratio = atr / data["close"]

        return vol_ratio > 0.002

    # -------------------------------------------------
    # Backtest Runner
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
