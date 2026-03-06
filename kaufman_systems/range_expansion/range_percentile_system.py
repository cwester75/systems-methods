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
