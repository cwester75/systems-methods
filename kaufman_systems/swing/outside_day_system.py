"""
Outside Day System

Concept
-------
An outside day (engulfing bar) occurs when the current bar's high
exceeds the previous bar's high AND the current bar's low is below the
previous bar's low.  The close direction determines the signal.

Signal logic
------------
Outside day + close > previous close → LONG
Outside day + close < previous close → SHORT
Otherwise                            → FLAT

Outside days represent a sudden expansion in volatility and often mark
short-term turning points or continuation accelerations.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OutsideDaySystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def is_outside_day(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < 2:
            return False

        return highs[-1] > highs[-2] and lows[-1] < lows[-2]

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
        closes = np.asarray(closes)

        if len(closes) < 2:
            return 0

        if not self.is_outside_day(highs, lows):
            return 0

        if closes[-1] > closes[-2]:
            return 1

        if closes[-1] < closes[-2]:
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
            "is_outside_day": self.is_outside_day(highs, lows),
        }
# swing/outside_day_system.py

import numpy as np
import pandas as pd


class OutsideDaySystem:
    """
    Outside Day Reversal System

    Concept
    -------
    An Outside Day occurs when:

        today's high > yesterday's high
        AND
        today's low < yesterday's low

    This indicates a volatility expansion and possible reversal.

    Trading Logic
    -------------

    Bullish Outside Day:
        outside day AND close > open
        -> go LONG next bar

    Bearish Outside Day:
        outside day AND close < open
        -> go SHORT next bar

    Exit:
        opposite signal or ATR stop
    """

    def __init__(
        self,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        max_hold: int = 10,
    ):
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade
        self.max_hold = max_hold

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
    # Signal Detection
    # -------------------------------------------------

    def detect_outside_days(self, data: pd.DataFrame):

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        outside_day = (data["high"] > prev_high) & (data["low"] < prev_low)

        bullish = outside_day & (data["close"] > data["open"])
        bearish = outside_day & (data["close"] < data["open"])

        return bullish, bearish

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        bullish, bearish = self.detect_outside_days(data)

        signal = pd.Series(index=data.index, dtype=float)

        signal[bullish] = 1
        signal[bearish] = -1

        signal = signal.shift(1)  # trade next bar
        signal = signal.ffill().fillna(0)

        # enforce maximum holding period
        holding = 0
        for i in range(len(signal)):
            if signal.iloc[i] != 0:
                holding += 1
                if holding > self.max_hold:
                    signal.iloc[i] = 0
                    holding = 0
            else:
                holding = 0

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

        mask = vol_ratio > 0.003

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
