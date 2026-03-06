# pattern/range_expansion_system.py

import numpy as np
import pandas as pd


class RangeExpansionSystem:
    """
    Range Expansion System

    Concept
    -------
    Markets often transition from low volatility to high volatility.
    When today's trading range expands significantly relative to recent
    history, it may signal the beginning of a directional move.

    Range = high - low

    A range expansion occurs when:

        today's_range > k * average_range

    Trading Logic
    -------------
    Bullish Expansion:
        close > open AND range expansion
        -> go LONG next bar

    Bearish Expansion:
        close < open AND range expansion
        -> go SHORT next bar

    Exit
    ----
    Opposite signal or ATR trailing stop
    """

    def __init__(
        self,
        range_length: int = 10,
        expansion_multiplier: float = 1.8,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        max_hold: int = 15,
    ):

        self.range_length = range_length
        self.expansion_multiplier = expansion_multiplier
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
    # Range Expansion Detection
    # -------------------------------------------------

    def detect_expansion(self, data: pd.DataFrame):

        range_today = data["high"] - data["low"]
        avg_range = range_today.rolling(self.range_length).mean()

        expansion = range_today > self.expansion_multiplier * avg_range

        bullish = expansion & (data["close"] > data["open"])
        bearish = expansion & (data["close"] < data["open"])

        return bullish, bearish

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        bullish, bearish = self.detect_expansion(data)

        signal = pd.Series(index=data.index, dtype=float)

        signal[bullish] = 1
        signal[bearish] = -1

        signal = signal.shift(1)
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
