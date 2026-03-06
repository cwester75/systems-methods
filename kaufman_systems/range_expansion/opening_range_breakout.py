"""
Opening Range Breakout System

Concept
-------
Uses the first N bars of a session to define the "opening range."
A breakout beyond this range signals a directional move.  Since we
work with daily bar data (no intraday), we approximate this as the
range of the first N bars from the start of the data window.

Signal logic
------------
If close > max(highs[:N]) → LONG
If close < min(lows[:N])  → SHORT
Otherwise                 → FLAT

In practice, the opening range is recalculated over the first N bars
of the lookback window, and the current close is compared to that range.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OpeningRangeBreakoutSystem(TradingSystem):
# range_expansion/opening_range_breakout.py

import numpy as np
import pandas as pd


class OpeningRangeBreakout:
    """
    Opening Range Breakout (ORB) System

    Concept
    -------
    The opening range represents the price discovery period
    immediately after market open. Breakouts from this range
    often produce directional intraday trends.

    Opening Range
    -------------
    Defined as the high and low of the first N bars of the session.

    Trading Logic
    -------------

    Long Entry:
        price breaks above opening range high

    Short Entry:
        price breaks below opening range low

    Exit:
        end of session or ATR stop
    """

    def __init__(
        self,
        opening_bars: int = 5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.opening_bars = opening_bars
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def opening_range(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.opening_bars + 1:
            return None, None

        or_high = np.max(highs[-self.opening_bars - 1:-1])
        or_low = np.min(lows[-self.opening_bars - 1:-1])

        return or_high, or_low

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

        or_high, or_low = self.opening_range(highs, lows)

        if or_high is None:
            return 0

        price = np.asarray(closes)[-1]

        if price > or_high:
            return 1

        if price < or_low:
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

        or_high, or_low = self.opening_range(highs, lows)

        return {
            "or_high": or_high,
            "or_low": or_low,
        }
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):

        self.opening_bars = opening_bars
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

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
    # Opening Range
    # -------------------------------------------------

    def compute_opening_range(self, data: pd.DataFrame):

        """
        Assumes intraday data with a DateTime index.
        """

        df = data.copy()
        df["date"] = df.index.date

        opening_high = []
        opening_low = []

        for d, group in df.groupby("date"):

            first = group.iloc[: self.opening_bars]

            high = first["high"].max()
            low = first["low"].min()

            opening_high.extend([high] * len(group))
            opening_low.extend([low] * len(group))

        df["opening_high"] = opening_high
        df["opening_low"] = opening_low

        return df["opening_high"], df["opening_low"]

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        opening_high, opening_low = self.compute_opening_range(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > opening_high
        short_entry = close < opening_low

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
