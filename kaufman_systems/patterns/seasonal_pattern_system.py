"""
Seasonal Pattern System
-----------------------

Concept
-------
Many assets exhibit recurring seasonal behavior.

Examples
--------
Retail stocks:
    strength into holidays

Energy markets:
    winter demand

Agricultural commodities:
    harvest cycles

This system computes historical seasonal returns and
trades when the current day-of-year has positive or
negative expected return.

Rules
-----
LONG
    expected seasonal return > threshold

SHORT
    expected seasonal return < -threshold
"""

import pandas as pd
import numpy as np


class SeasonalPatternSystem:

    def __init__(
        self,
        lookback_years: int = 10,
        threshold: float = 0.002,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        lookback_years : int
            number of years used for seasonal statistics
        threshold : float
            minimum seasonal bias to trigger trade
        risk_per_trade : float
            portfolio risk fraction
        """

        self.lookback_years = lookback_years
        self.threshold = threshold
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Seasonal return profile
    # ---------------------------------------------------------

    def seasonal_returns(self, df: pd.DataFrame):

        data = df.copy()

        data["returns"] = data["close"].pct_change()
        data["day_of_year"] = data.index.dayofyear

        # restrict to last N years
        cutoff = data.index.max() - pd.DateOffset(years=self.lookback_years)
        data = data[data.index >= cutoff]

        seasonal_mean = data.groupby("day_of_year")["returns"].mean()

        return seasonal_mean


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

        seasonal_profile = self.seasonal_returns(df)

        today = df.index[-1].dayofyear

        expected_return = seasonal_profile.get(today, 0)

        if expected_return > self.threshold:
            return 1

        if expected_return < -self.threshold:
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
        """
        Liquidity filter
        """

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
