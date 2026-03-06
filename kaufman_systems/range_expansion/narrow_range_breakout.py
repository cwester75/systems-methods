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
