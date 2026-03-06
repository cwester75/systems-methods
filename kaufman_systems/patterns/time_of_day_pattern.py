"""
Time-of-Day Pattern System
--------------------------

Concept
-------
Markets exhibit recurring intraday behavior patterns:

Opening Session
    high volatility
Midday
    consolidation
Closing Session
    directional move / position squaring

This system trades breakouts during the opening window
and optionally exits during the closing session.

Data Requirement
----------------
Intraday bars with a DateTime index.
"""

import pandas as pd
import numpy as np


class TimeOfDayPatternSystem:

    def __init__(
        self,
        open_window_minutes: int = 30,
        close_window_minutes: int = 30,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        open_window_minutes : int
            opening range measurement window
        close_window_minutes : int
            closing exit window
        risk_per_trade : float
            portfolio risk per trade
        """

        self.open_window_minutes = open_window_minutes
        self.close_window_minutes = close_window_minutes
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Opening range calculation
    # ---------------------------------------------------------

    def opening_range(self, df: pd.DataFrame):

        session_start = df.index[0]

        window_end = session_start + pd.Timedelta(
            minutes=self.open_window_minutes
        )

        opening_data = df[df.index <= window_end]

        high = opening_data["high"].max()
        low = opening_data["low"].min()

        return high, low


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Returns
        -------
        1  -> long breakout
        -1 -> short breakout
        0  -> no signal
        """

        high, low = self.opening_range(df)

        price = df["close"].iloc[-1]

        if price > high:
            return 1

        if price < low:
            return -1

        return 0


    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(
        self,
        capital: float,
        volatility: float
    ) -> float:

        risk_dollars = capital * self.risk_per_trade

        if volatility == 0:
            return 0

        position = risk_dollars / volatility

        return position


    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, df: pd.DataFrame) -> bool:

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
