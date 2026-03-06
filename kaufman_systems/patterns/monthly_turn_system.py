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

import numpy as np

from kaufman_systems.base import TradingSystem


class MonthlyTurnSystem(TradingSystem):

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
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``day_of_month`` (int, 1-based position
        within the trading month) and ``trading_days_remaining`` (int,
        trading days left in the month including today).

        Returns
        -------
        1 -> long
        0 -> flat
        """
        day_of_month = data["day_of_month"]
        remaining = data["trading_days_remaining"]

        if remaining <= self.pre_days:
            return 1

        if day_of_month <= self.post_days:
            return 1

        return 0

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
