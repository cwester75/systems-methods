"""
Holiday Effect Pattern System
-----------------------------

Concept
-------
Equity markets often experience positive returns immediately
before major holidays (e.g., Christmas, Thanksgiving).

Rules
-----
LONG
    enter at close before holiday

EXIT
    next trading day close
"""

import pandas as pd


class HolidayEffectSystem:

    def __init__(
        self,
        holidays=None,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        holidays : list of datetime.date
            list of market holiday dates
        risk_per_trade : float
            portfolio risk fraction
        """

        self.holidays = holidays if holidays is not None else []
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Determine if tomorrow is a holiday
    # ---------------------------------------------------------

    def tomorrow_is_holiday(self, df: pd.DataFrame) -> bool:

        today = df.index[-1].date()

        for holiday in self.holidays:
            if (holiday - today).days == 1:
                return True

        return False


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Returns
        -------
        1 -> long
        -1 -> exit
        0 -> no signal
        """

        today = df.index[-1].date()

        if self.tomorrow_is_holiday(df):
            return 1

        if today in self.holidays:
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

        return risk_dollars / volatility


    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, df: pd.DataFrame) -> bool:

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
