"""
Weekend Effect Pattern System
-----------------------------

Concept
-------
Certain markets historically show abnormal return patterns across weekends.

Typical anomaly (varies by asset):
    Friday close -> Monday close

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

import numpy as np

from kaufman_systems.base import TradingSystem


class WeekendEffectSystem(TradingSystem):

    def __init__(
        self,
        direction: str = "long",
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        direction : str
            'long'  -> Friday close -> Monday close
            'short' -> Monday close -> Friday close
        risk_per_trade : float
            portfolio risk fraction
        """
        self.direction = direction
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include a ``weekday`` key (int, Monday=0 ... Friday=4).

        Returns
        -------
        1  -> long
        -1 -> short
        0  -> no trade
        """
        weekday = data["weekday"]

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

    def position_sizing(self, data: dict, risk: dict) -> float:
        """Volatility-based sizing."""
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
        """Liquidity filter — requires volume above 20-day average."""
        volumes = np.asarray(data["volumes"], dtype=float)

        if len(volumes) < 20:
            return False

        avg_volume = np.mean(volumes[-20:])
        return float(volumes[-1]) > avg_volume
