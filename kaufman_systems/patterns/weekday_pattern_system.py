"""
Weekday Pattern System
----------------------

Concept
-------
Some assets show consistent return bias by weekday.

Example patterns often observed:
    Monday      weak
    Wednesday   strong
    Friday      profit taking

The system measures rolling average weekday returns and
trades when the current weekday shows statistically positive
or negative bias.

Rules
-----
LONG
    if weekday average return > threshold

SHORT
    if weekday average return < -threshold
"""

import pandas as pd
import numpy as np


class WeekdayPatternSystem:

    def __init__(
        self,
        lookback: int = 252,
        threshold: float = 0.001,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        lookback : int
            number of days used for weekday statistics
        threshold : float
            minimum expected return to trigger signal
        risk_per_trade : float
            portfolio risk fraction
        """

        self.lookback = lookback
        self.threshold = threshold
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Calculate weekday return profile
    # ---------------------------------------------------------

    def weekday_returns(self, df: pd.DataFrame):

        data = df.copy()

        data["returns"] = data["close"].pct_change()
        data["weekday"] = data.index.weekday

        sample = data.tail(self.lookback)

        weekday_mean = sample.groupby("weekday")["returns"].mean()

        return weekday_mean


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

        weekday_profile = self.weekday_returns(df)

        today = df.index[-1].weekday()

        expected_return = weekday_profile.get(today, 0)

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
