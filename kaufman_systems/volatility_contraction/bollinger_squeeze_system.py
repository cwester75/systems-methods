"""
Bollinger Squeeze System

Concept
-------
Detects Bollinger Band contraction (squeeze) as a precursor to
volatility expansion.  When the bandwidth (distance between upper and
lower bands relative to the MA) falls below a threshold, the market is
in a squeeze.  The breakout direction after the squeeze fires the signal.

Signal logic
------------
If bandwidth < squeeze_threshold:
  Close > Upper Band → LONG
  Close < Lower Band → SHORT
Otherwise            → FLAT

Kaufman discusses Bollinger squeezes as low-risk entry setups where
compressed volatility precedes directional moves.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class BollingerSqueezeSystem(TradingSystem):

    def __init__(
        self,
        ma_period: int = 20,
        band_mult: float = 2.0,
        squeeze_threshold: float = 0.04,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.ma_period = ma_period
        self.band_mult = band_mult
        self.squeeze_threshold = squeeze_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def bollinger(self, prices):
        """Compute bands from bars *prior* to the current bar.

        The squeeze is a condition on the recent past; the current bar
        is the one that breaks out.  Using bars[:-1] prevents the
        breakout bar from inflating the bandwidth calculation.
        """
        prices = np.asarray(prices)

        if len(prices) < self.ma_period + 1:
            return None, None, None, None

        window = prices[-(self.ma_period + 1):-1]
        ma = np.mean(window)
        sd = np.std(window)

        upper = ma + self.band_mult * sd
        lower = ma - self.band_mult * sd
        bandwidth = (upper - lower) / ma if ma != 0 else None

        return ma, upper, lower, bandwidth

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

        ma, upper, lower, bandwidth = self.bollinger(closes)

        if bandwidth is None:
            return 0

        if bandwidth >= self.squeeze_threshold:
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

        ma, upper, lower, bandwidth = self.bollinger(closes)

        return {
            "ma": ma,
            "upper_band": upper,
            "lower_band": lower,
            "bandwidth": bandwidth,
            "is_squeeze": bandwidth is not None and bandwidth < self.squeeze_threshold,
        }
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
