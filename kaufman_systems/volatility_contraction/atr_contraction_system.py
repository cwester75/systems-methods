# volatility_contraction/atr_contraction_system.py

import numpy as np
import pandas as pd


class ATRContractionSystem:
    """
    ATR Contraction Breakout System

    Concept
    -------
    Volatility contraction frequently precedes large directional moves.
    This system detects unusually low ATR relative to its historical
    distribution and trades the breakout.

    Contraction Condition
    ---------------------
        ATR < rolling percentile threshold

    Breakout
    --------
    Long:
        close > highest high of contraction window

    Short:
        close < lowest low of contraction window
    """

    def __init__(
        self,
        atr_length: int = 14,
        contraction_lookback: int = 100,
        contraction_percentile: float = 0.2,
        breakout_length: int = 20,
        atr_stop_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.atr_length = atr_length
        self.contraction_lookback = contraction_lookback
        self.contraction_percentile = contraction_percentile
        self.breakout_length = breakout_length
        self.atr_stop_multiplier = atr_stop_multiplier
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
    # Contraction Detection
    # -------------------------------------------------

    def detect_contraction(self, atr: pd.Series):

        threshold = atr.rolling(self.contraction_lookback).quantile(
            self.contraction_percentile
        )

        contraction = atr < threshold

        return contraction

    # -------------------------------------------------
    # Breakout Levels
    # -------------------------------------------------

    def breakout_levels(self, data: pd.DataFrame):

        upper = data["high"].rolling(self.breakout_length).max()
        lower = data["low"].rolling(self.breakout_length).min()

        return upper, lower

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        atr = self.atr(data)
        contraction = self.detect_contraction(atr)

        upper, lower = self.breakout_levels(data)

        close = data["close"]

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (close > upper.shift(1)) & contraction.shift(1)
        short_entry = (close < lower.shift(1)) & contraction.shift(1)

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

        position_size = risk_dollars / (atr * self.atr_stop_multiplier)

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
