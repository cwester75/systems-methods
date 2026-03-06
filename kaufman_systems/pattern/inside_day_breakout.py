"""
Inside Day Breakout System

Concept
-------
An inside day occurs when the current bar's range is entirely
contained within the previous bar's range (lower high, higher low).
This signals compression.  A breakout from the mother bar's range
on the next bar triggers a directional signal.

Signal logic
------------
If the previous bar was an inside day (bar[-2] contained within bar[-3]):
  Close > high of mother bar (bar[-3]) → LONG
  Close < low of mother bar  (bar[-3]) → SHORT
Otherwise                              → FLAT

We look for an inside day at bar[-2] relative to bar[-3] (the "mother"
bar), then check if the current bar[-1] breaks out of the mother bar's
range.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class InsideDayBreakoutSystem(TradingSystem):

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

    def inside_day_setup(self, highs, lows):
        """Check if bar[-2] is inside bar[-3] and return mother bar levels."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < 3:
            return False, None, None

        mother_high = highs[-3]
        mother_low = lows[-3]
        inside_high = highs[-2]
        inside_low = lows[-2]

        is_inside = inside_high <= mother_high and inside_low >= mother_low

        if is_inside:
            return True, mother_high, mother_low

        return False, None, None

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

        is_setup, mother_high, mother_low = self.inside_day_setup(highs, lows)

        if not is_setup:
            return 0

        price = closes[-1]

        if price > mother_high:
            return 1

        if price < mother_low:
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

        is_setup, mother_high, mother_low = self.inside_day_setup(highs, lows)

        return {
            "is_inside_day_setup": is_setup,
            "mother_high": mother_high,
            "mother_low": mother_low,
        }
# pattern/inside_day_breakout.py

import numpy as np
import pandas as pd


class InsideDayBreakoutSystem:
    """
    Inside Day Breakout System

    Concept
    -------
    An inside day occurs when:

        today's high < yesterday's high
        AND
        today's low > yesterday's low

    This indicates volatility compression. A breakout of the
    inside-day range often leads to directional movement.

    Trading Logic
    -------------
    Long:
        price breaks above inside-day high

    Short:
        price breaks below inside-day low

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
    # Inside Day Detection
    # -------------------------------------------------

    def detect_inside_day(self, data: pd.DataFrame):

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        inside_day = (data["high"] < prev_high) & (data["low"] > prev_low)

        inside_high = data["high"]
        inside_low = data["low"]

        return inside_day, inside_high, inside_low

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        inside_day, inside_high, inside_low = self.detect_inside_day(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_breakout = (close > inside_high.shift(1)) & inside_day.shift(1)
        short_breakout = (close < inside_low.shift(1)) & inside_day.shift(1)

        signal[long_breakout] = 1
        signal[short_breakout] = -1

        signal = signal.ffill().fillna(0)

        # enforce max holding period
        hold = 0
        for i in range(len(signal)):
            if signal.iloc[i] != 0:
                hold += 1
                if hold > self.max_hold:
                    signal.iloc[i] = 0
                    hold = 0
            else:
                hold = 0

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
