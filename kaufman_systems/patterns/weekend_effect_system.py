"""
Weekend Effect Pattern System
-----------------------------

Concept
-------
Certain markets historically show abnormal return patterns across weekends.

Typical anomaly (varies by asset):
    Friday close → Monday close

Rules
-----
LONG
    Enter Friday close
    Exit Monday close

SHORT (optional variant)
    Enter Monday close
    Exit Friday close

Filters
-------
Optional trend and volatility filters can be applied.
"""

import pandas as pd
import numpy as np


class WeekendEffectSystem:

    def __init__(
        self,
        direction: str = "long",
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        direction : str
            'long'  -> Friday close → Monday close
            'short' -> Monday close → Friday close
        risk_per_trade : float
            portfolio risk fraction
        """
        self.direction = direction
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Generate trading signal.

        Returns
        -------
        1  -> long
        -1 -> short
        0  -> no trade
        """

        today = df.index[-1]
        weekday = today.weekday()  # Monday=0 ... Friday=4

        if self.direction == "long":

            if weekday == 4:   # Friday
                return 1

            if weekday == 0:   # Monday
                return -1

        elif self.direction == "short":

            if weekday == 0:
                return -1

            if weekday == 4:
                return 1

        return 0


    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(
        self,
        capital: float,
        volatility: float
    ) -> float:
        """
        Volatility-based sizing
        """

        risk_dollars = capital * self.risk_per_trade

        if volatility == 0:
            return 0

        position = risk_dollars / volatility

        return position


    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, df: pd.DataFrame) -> bool:
        """
        Optional liquidity filter
        """

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
