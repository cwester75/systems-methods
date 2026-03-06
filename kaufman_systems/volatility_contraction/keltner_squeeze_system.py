# volatility_contraction/keltner_squeeze_system.py

import numpy as np
import pandas as pd


class KeltnerSqueezeSystem:
    """
    Keltner Channel Squeeze System

    Concept
    -------
    Detect volatility compression using Keltner Channels.
    A squeeze occurs when the channel width becomes unusually narrow.

    Keltner Channels
    ----------------
        middle = EMA(close, n)
        upper  = middle + k * ATR
        lower  = middle - k * ATR

    Channel Width
    -------------
        width = (upper - lower) / middle

    Trading Logic
    -------------
    Long Entry:
        squeeze active AND close > upper channel

    Short Entry:
        squeeze active AND close < lower channel

    Exit:
        price crosses middle channel or opposite signal
    """

    def __init__(
        self,
        ema_length: int = 20,
        atr_length: int = 14,
        atr_multiplier: float = 1.5,
        squeeze_lookback: int = 100,
        squeeze_percentile: float = 0.2,
        risk_per_trade: float = 0.01,
    ):

        self.ema_length = ema_length
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # EMA
    # -------------------------------------------------

    def ema(self, series: pd.Series):

        return series.ewm(span=self.ema_length, adjust=False).mean()

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
    # Keltner Channel
    # -------------------------------------------------

    def keltner_channel(self, data: pd.DataFrame):

        middle = self.ema(data["close"])
        atr = self.atr(data)

        upper = middle + self.atr_multiplier * atr
        lower = middle - self.atr_multiplier * atr

        width = (upper - lower) / middle

        return middle, upper, lower, width

    # -------------------------------------------------
    # Squeeze Detection
    # -------------------------------------------------

    def detect_squeeze(self, width: pd.Series):

        threshold = width.rolling(self.squeeze_lookback).quantile(
            self.squeeze_percentile
        )

        squeeze = width < threshold

        return squeeze

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        middle, upper, lower, width = self.keltner_channel(data)

        squeeze = self.detect_squeeze(width)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (close > upper.shift(1)) & squeeze.shift(1)
        short_entry = (close < lower.shift(1)) & squeeze.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit conditions
        exit_long = close < middle.shift(1)
        exit_short = close > middle.shift(1)

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

        return vol_ratio > 0.003

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
