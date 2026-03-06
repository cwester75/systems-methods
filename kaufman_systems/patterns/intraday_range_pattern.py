"""
Intraday Range Pattern System
-----------------------------

Concept
-------
The early session often establishes a directional range.
Breakouts beyond this range frequently lead to trend moves.

Rules
-----
LONG
    price breaks above opening range high

SHORT
    price breaks below opening range low

Exit
    end of session or risk stop
"""

import pandas as pd
import numpy as np


class IntradayRangePatternSystem:

    def __init__(
        self,
        range_minutes: int = 30,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        range_minutes : int
            time window used to define opening range
        risk_per_trade : float
            portfolio risk fraction
        """

        self.range_minutes = range_minutes
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Opening range calculation
    # ---------------------------------------------------------

    def opening_range(self, df: pd.DataFrame):

        session_start = df.index[0]

        range_end = session_start + pd.Timedelta(
            minutes=self.range_minutes
        )

        range_data = df[df.index <= range_end]

        range_high = range_data["high"].max()
        range_low = range_data["low"].min()

        return range_high, range_low


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

        range_high, range_low = self.opening_range(df)

        price = df["close"].iloc[-1]

        if price > range_high:
            return 1

        if price < range_low:
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
