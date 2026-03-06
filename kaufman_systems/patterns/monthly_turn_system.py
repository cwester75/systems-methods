"""
Monthly Turn (Turn-of-the-Month) Pattern System
-----------------------------------------------

Concept
-------
Equity markets tend to produce abnormal positive returns
around the end of each month.

Typical window:
    T-2  -> T+3

Where T = last trading day of the month.

Rules
-----
LONG
    enter within TOM window

EXIT
    after window closes
"""

import pandas as pd
import numpy as np


class MonthlyTurnSystem:

    def __init__(
        self,
        pre_days: int = 2,
        post_days: int = 3,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        pre_days : int
            trading days before month end
        post_days : int
            trading days after month start
        risk_per_trade : float
            portfolio risk fraction
        """

        self.pre_days = pre_days
        self.post_days = post_days
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Determine trading-day position within month
    # ---------------------------------------------------------

    def month_position(self, df: pd.DataFrame):

        dates = df.index

        current = dates[-1]

        month_data = df[df.index.month == current.month]

        position = month_data.index.get_loc(current)

        remaining = len(month_data) - position - 1

        return position, remaining


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Returns
        -------
        1 -> long
        0 -> flat
        """

        position, remaining = self.month_position(df)

        if remaining <= self.pre_days:
            return 1

        if position <= self.post_days:
            return 1

        return 0


    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(self, capital, volatility):

        risk_dollars = capital * self.risk_per_trade

        if volatility == 0:
            return 0

        return risk_dollars / volatility


    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, df):

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
