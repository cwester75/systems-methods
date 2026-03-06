"""
Narrow Range Breakout System (NR4/NR7)

Concept
-------
Identifies bars with the narrowest range in N periods (classic NR4 or
NR7 patterns).  A narrow-range bar signals compression; a breakout
from that bar's high or low on the next bar triggers a signal.

Signal logic
------------
If bar[-2] had the narrowest range in the last N bars:
  Close > high of bar[-2] → LONG
  Close < low of bar[-2]  → SHORT
Otherwise                 → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class NarrowRangeBreakoutSystem(TradingSystem):

    def __init__(
        self,
        nr_period: int = 7,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.nr_period = nr_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def is_narrow_range(self, highs, lows):
        """Check if bar[-2] has the narrowest range in the last nr_period bars."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.nr_period + 1:
            return False, None, None

        ranges = highs - lows
        lookback_ranges = ranges[-(self.nr_period + 1):-1]
        nr_bar_range = ranges[-2]

        if nr_bar_range == np.min(lookback_ranges):
            return True, highs[-2], lows[-2]

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

        is_nr, nr_high, nr_low = self.is_narrow_range(highs, lows)

        if not is_nr:
            return 0

        price = closes[-1]

        if price > nr_high:
            return 1

        if price < nr_low:
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

        is_nr, nr_high, nr_low = self.is_narrow_range(highs, lows)

        return {
            "is_narrow_range": is_nr,
            "nr_high": nr_high,
            "nr_low": nr_low,
        }
# range_expansion/narrow_range_breakout.py

import numpy as np
import pandas as pd


class NarrowRangeBreakout:
    """
    Narrow Range Breakout System (NR4 / NR7 style)

    Concept
    -------
    A Narrow Range day occurs when today's trading range
    is the smallest of the last N days.

    Range = high - low

    A volatility compression event often precedes a
    directional expansion.

    Trading Logic
    -------------

    Detect NRN (Narrowest Range in N days)

    Long Entry:
        close > NR high

    Short Entry:
        close < NR low

    Exit:
        opposite breakout or ATR stop
    """

    def __init__(
        self,
        narrow_period: int = 7,      # NR7 by default
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        max_hold: int = 10,
    ):

        self.narrow_period = narrow_period
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
    # Narrow Range Detection
    # -------------------------------------------------

    def detect_narrow_range(self, data: pd.DataFrame):

        ranges = data["high"] - data["low"]

        rolling_min = ranges.rolling(self.narrow_period).min()

        narrow_day = ranges == rolling_min

        return narrow_day

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        narrow_day = self.detect_narrow_range(data)

        nr_high = data["high"]
        nr_low = data["low"]

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (close > nr_high.shift(1)) & narrow_day.shift(1)
        short_entry = (close < nr_low.shift(1)) & narrow_day.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # enforce maximum holding period
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
