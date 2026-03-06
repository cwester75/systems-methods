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

import numpy as np

from kaufman_systems.base import TradingSystem


class HolidayEffectSystem(TradingSystem):

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
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``today`` (datetime.date) and
        ``tomorrow_is_holiday`` (bool).

        Returns
        -------
        1 -> long
        -1 -> exit
        0 -> no signal
        """
        today = data["today"]
        tomorrow_is_holiday = data.get("tomorrow_is_holiday", False)

        if tomorrow_is_holiday:
            return 1

        if today in self.holidays:
            return -1

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
