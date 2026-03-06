"""
Reversal Pattern System
-----------------------

Concept
-------
Reversal bars indicate exhaustion of the current move.

Common patterns detected:
    - Key Reversal Day
    - Outside Reversal Bar

Bullish Key Reversal
    new low + strong close above prior close

Bearish Key Reversal
    new high + strong close below prior close
"""

import pandas as pd
import numpy as np


class ReversalPatternSystem:

    def __init__(
        self,
        lookback: int = 20,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        lookback : int
            period used to determine local extremes
        risk_per_trade : float
            portfolio risk per trade
        """

        self.lookback = lookback
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Detect bullish reversal
    # ---------------------------------------------------------

    def bullish_reversal(self, df):

        today = df.iloc[-1]
        prev = df.iloc[-2]

        lowest = df["low"].rolling(self.lookback).min().iloc[-2]

        condition1 = today["low"] < lowest
        condition2 = today["close"] > prev["close"]

        return condition1 and condition2


    # ---------------------------------------------------------
    # Detect bearish reversal
    # ---------------------------------------------------------

    def bearish_reversal(self, df):

        today = df.iloc[-1]
        prev = df.iloc[-2]

        highest = df["high"].rolling(self.lookback).max().iloc[-2]

        condition1 = today["high"] > highest
        condition2 = today["close"] < prev["close"]

        return condition1 and condition2


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Returns
        -------
        1  -> long
        -1 -> short
        0  -> neutral
        """

        if self.bullish_reversal(df):
            return 1

        if self.bearish_reversal(df):
            return -1

        return 0


    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(self, capital, atr):

        risk_dollars = capital * self.risk_per_trade

        if atr == 0:
            return 0

        return risk_dollars / atr


    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, df):

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
