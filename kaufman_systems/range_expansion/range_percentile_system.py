"""
Range Percentile System

Concept
-------
Ranks the current bar's range against the distribution of ranges over
the last N bars.  When the current range falls in an extreme percentile,
it signals either compression (ready to expand) or expansion (breakout
in progress).

Signal logic
------------
If range percentile > upper_pct (e.g., 90th percentile — expansion):
  Close > prev close → LONG
  Close < prev close → SHORT
Otherwise            → FLAT

Kaufman discusses percentile-based filters as a way to rank current
volatility relative to recent history.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class RangePercentileSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 50,
        upper_pct: float = 90.0,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.upper_pct = upper_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def range_percentile(self, highs, lows):
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.lookback + 1:
            return None

        ranges = highs - lows
        current_range = ranges[-1]
        historical_ranges = ranges[-(self.lookback + 1):-1]

        percentile = np.sum(historical_ranges <= current_range) / len(historical_ranges) * 100

        return percentile

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

        pct = self.range_percentile(highs, lows)

        if pct is None:
            return 0

        if pct <= self.upper_pct:
            return 0

        if len(closes) < 2:
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
            "range_percentile": self.range_percentile(highs, lows),
            "upper_pct": self.upper_pct,
        }
# range_expansion/range_percentile_system.py

import numpy as np
import pandas as pd


class RangePercentileSystem:
    """
    Range Percentile Breakout System

    Concept
    -------
    Measures the current trading range relative to the historical
    distribution of ranges. When the range moves into the upper
    percentile of its historical distribution, volatility expansion
    may signal the beginning of a trend.

    Range = high - low

    Range Percentile
    ----------------
    percentile = percentile_rank(current_range, lookback_ranges)

    Trading Logic
    -------------
    Long Entry:
        range_percentile > expansion_percentile AND
        close > previous high

    Short Entry:
        range_percentile > expansion_percentile AND
        close < previous low

    Exit:
        range_percentile falls below exit_percentile
    """

    def __init__(
        self,
        percentile_lookback: int = 100,
        expansion_percentile: float = 0.85,
        exit_percentile: float = 0.50,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.percentile_lookback = percentile_lookback
        self.expansion_percentile = expansion_percentile
        self.exit_percentile = exit_percentile
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
    # Range Percentile
    # -------------------------------------------------

    def range_percentile(self, data: pd.DataFrame):

        ranges = data["high"] - data["low"]

        def percentile_rank(window):
            last = window[-1]
            return np.sum(window <= last) / len(window)

        percentile = ranges.rolling(self.percentile_lookback).apply(
            percentile_rank, raw=True
        )

        return percentile

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        rp = self.range_percentile(data)

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (rp > self.expansion_percentile) & (close > prev_high)
        short_entry = (rp > self.expansion_percentile) & (close < prev_low)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        exit_condition = rp < self.exit_percentile
        signal[exit_condition] = 0

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
