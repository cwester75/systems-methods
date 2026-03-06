"""
Earnings Drift Pattern System
-----------------------------

Concept
-------
Stocks often continue trending after earnings announcements
when the earnings surprise is significant.

Rules
-----
LONG
    earnings_surprise > positive_threshold

SHORT
    earnings_surprise < negative_threshold

EXIT
    after holding_period days
"""

import pandas as pd


class EarningsDriftSystem:

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        holding_period: int = 10,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        positive_threshold : float
            positive earnings surprise threshold
        negative_threshold : float
            negative earnings surprise threshold
        holding_period : int
            number of days to hold trade
        risk_per_trade : float
            portfolio risk fraction
        """

        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.holding_period = holding_period
        self.risk_per_trade = risk_per_trade


    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, earnings_surprise: float) -> int:
        """
        Returns
        -------
        1  -> long
        -1 -> short
        0  -> no trade
        """

        if earnings_surprise > self.positive_threshold:
            return 1

        if earnings_surprise < self.negative_threshold:
            return -1

        return 0


    # ---------------------------------------------------------
    # Exit rule
    # ---------------------------------------------------------

    def exit_signal(self, entry_index: int, current_index: int) -> bool:
        """
        Exit after fixed holding period
        """

        return (current_index - entry_index) >= self.holding_period


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

    def risk_filter(self, df: pd.DataFrame):

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
