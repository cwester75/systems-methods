# pattern/congestion_breakout_system.py

import numpy as np
import pandas as pd


class CongestionBreakoutSystem:
    """
    Congestion Breakout System

    Concept
    -------
    Markets often pause in a narrow consolidation ("congestion") before
    making a directional move. This system identifies low-range periods
    and trades the breakout.

    Congestion Detection
    --------------------
    A congestion zone occurs when:

        rolling_range < congestion_threshold

    where:
        rolling_range = (highest_high - lowest_low) / price

    Breakout
    --------
    Long:
        price breaks above congestion high

    Short:
        price breaks below congestion low
    """

    def __init__(
        self,
        congestion_length: int = 10,
        congestion_threshold: float = 0.02,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.congestion_length = congestion_length
        self.congestion_threshold = congestion_threshold
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
    # Congestion Detection
    # -------------------------------------------------

    def congestion_zone(self, data: pd.DataFrame):

        highest = data["high"].rolling(self.congestion_length).max()
        lowest = data["low"].rolling(self.congestion_length).min()

        range_pct = (highest - lowest) / data["close"]

        congestion = range_pct < self.congestion_threshold

        return congestion, highest, lowest

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        congestion, highest, lowest = self.congestion_zone(data)

        signal = pd.Series(index=data.index, dtype=float)

        breakout_long = (close > highest.shift(1)) & congestion.shift(1)
        breakout_short = (close < lowest.shift(1)) & congestion.shift(1)

        signal[breakout_long] = 1
        signal[breakout_short] = -1

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
