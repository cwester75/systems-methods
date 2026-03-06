"""
Keltner Squeeze System

Concept
-------
Detects when Bollinger Bands contract inside Keltner Channels, signaling
extreme volatility compression.  The squeeze fires when the Bollinger
upper band drops below the Keltner upper channel and the Bollinger lower
band rises above the Keltner lower channel.  Breakout direction on
release determines the signal.

Signal logic
------------
If Bollinger bands inside Keltner channels (squeeze active):
  Close > Keltner upper → LONG
  Close < Keltner lower → SHORT
Otherwise               → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class KeltnerSqueezeSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        bb_mult: float = 2.0,
        kc_mult: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.bb_mult = bb_mult
        self.kc_mult = kc_mult
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def bollinger_bands(self, prices):
        prices = np.asarray(prices)

        if len(prices) < self.ma_period:
            return None, None, None

        window = prices[-self.ma_period:]
        ma = np.mean(window)
        sd = np.std(window)

        return ma, ma + self.bb_mult * sd, ma - self.bb_mult * sd

    def keltner_channels(self, highs, lows, closes):
        closes = np.asarray(closes)

        if len(closes) < self.ma_period:
            return None, None, None

        ma = np.mean(closes[-self.ma_period:])
        atr = self.atr(highs, lows, closes)

        if atr is None:
            return None, None, None

        return ma, ma + self.kc_mult * atr, ma - self.kc_mult * atr

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

    def is_squeeze(self, highs, lows, closes):
        _, bb_upper, bb_lower = self.bollinger_bands(closes)
        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        if bb_upper is None or kc_upper is None:
            return False

        return bb_upper < kc_upper and bb_lower > kc_lower

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        if not self.is_squeeze(highs, lows, closes):
            return 0

        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        price = np.asarray(closes)[-1]

        if price > kc_upper:
            return 1

        if price < kc_lower:
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
        closes = data["closes"]

        _, bb_upper, bb_lower = self.bollinger_bands(closes)
        _, kc_upper, kc_lower = self.keltner_channels(highs, lows, closes)

        return {
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
            "is_squeeze": self.is_squeeze(highs, lows, closes),
        }
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
