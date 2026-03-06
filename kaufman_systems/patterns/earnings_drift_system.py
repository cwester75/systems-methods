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

import numpy as np

from kaufman_systems.base import TradingSystem


class EarningsDriftSystem(TradingSystem):

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

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``earnings_surprise`` (float).

        Returns
        -------
        1  -> long
        -1 -> short
        0  -> no trade
        """
        earnings_surprise = data["earnings_surprise"]

        if earnings_surprise > self.positive_threshold:
            return 1

        if earnings_surprise < self.negative_threshold:
            return -1

        return 0

    # ---------------------------------------------------------
    # Exit rule
    # ---------------------------------------------------------

    def exit_signal(self, entry_index: int, current_index: int) -> bool:
        """Exit after fixed holding period."""
        return (current_index - entry_index) >= self.holding_period

    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(self, data: dict, risk: dict) -> float:
        closes = np.asarray(data["closes"], dtype=float)
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        if len(closes) < 2:
            return 0

        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns[-20:]) * closes[-1] if len(returns) >= 20 else 0

        if volatility == 0:
            return 0

        return equity * risk_per_trade / volatility

    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, data: dict) -> bool:
        volumes = np.asarray(data["volumes"], dtype=float)

        if len(volumes) < 20:
            return False

        avg_volume = np.mean(volumes[-20:])
        return float(volumes[-1]) > avg_volume
