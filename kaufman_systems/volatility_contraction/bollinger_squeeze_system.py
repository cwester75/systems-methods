# volatility_contraction/bollinger_squeeze_system.py

import numpy as np
import pandas as pd


class BollingerSqueezeSystem:
    """
    Bollinger Band Squeeze System

    Concept
    -------
    Volatility contraction often precedes volatility expansion.
    A Bollinger "squeeze" occurs when the Bollinger Band width
    falls below a historical percentile.

    Bollinger Bands:
        middle = SMA(close, n)
        upper  = middle + k * std
        lower  = middle - k * std

    Band Width:
        (upper - lower) / middle

    Trading Logic
    -------------

    Squeeze Condition:
        band_width < rolling percentile threshold

    Long Entry:
        squeeze active AND close > upper band

    Short Entry:
        squeeze active AND close < lower band

    Exit:
        opposite signal or trailing ATR stop
    """

    def __init__(
        self,
        bb_length: int = 20,
        bb_std: float = 2.0,
        squeeze_lookback: int = 100,
        squeeze_percentile: float = 0.2,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):

        self.bb_length = bb_length
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------

    def bollinger_bands(self, data: pd.DataFrame):

        close = data["close"]

        middle = close.rolling(self.bb_length).mean()
        std = close.rolling(self.bb_length).std()

        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std

        width = (upper - lower) / middle

        return middle, upper, lower, width

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
    # Squeeze Detection
    # -------------------------------------------------

    def detect_squeeze(self, width: pd.Series):

        rolling_threshold = width.rolling(self.squeeze_lookback).quantile(
            self.squeeze_percentile
        )

        squeeze = width < rolling_threshold

        return squeeze

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        middle, upper, lower, width = self.bollinger_bands(data)

        squeeze = self.detect_squeeze(width)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (close > upper.shift(1)) & squeeze.shift(1)
        short_entry = (close < lower.shift(1)) & squeeze.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

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

        return vol_ratio > 0.003

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
