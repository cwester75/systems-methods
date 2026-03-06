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

import numpy as np

from kaufman_systems.base import TradingSystem


class SeasonalPatternSystem(TradingSystem):

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

    def _seasonal_returns(self, closes, day_of_years):
        """Compute mean return per day-of-year."""
        closes = np.asarray(closes, dtype=float)
        day_of_years = np.asarray(day_of_years, dtype=int)

        returns = np.diff(closes) / closes[:-1]
        doys = day_of_years[1:]

        means = {}
        for d in np.unique(doys):
            mask = doys == d
            if np.any(mask):
                means[int(d)] = float(np.mean(returns[mask]))
        return means

    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``closes``, ``day_of_years``
        (array of int 1–366), and ``day_of_year`` (int, current day).

        Returns
        -------
        1  -> long
        -1 -> short
        0  -> neutral
        """
        seasonal_profile = self._seasonal_returns(
            data["closes"], data["day_of_years"]
        )
        today = data["day_of_year"]

        expected_return = seasonal_profile.get(today, 0)

        if expected_return > self.threshold:
            return 1

        if expected_return < -self.threshold:
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
        """Liquidity filter."""
        volumes = np.asarray(data["volumes"], dtype=float)

        if len(volumes) < 20:
            return False

        avg_volume = np.mean(volumes[-20:])
        return float(volumes[-1]) > avg_volume

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def indicators(self, data: dict) -> dict:
        profile = self._seasonal_returns(data["closes"], data["day_of_years"])
        today = data["day_of_year"]
        return {
            "seasonal_returns": profile,
            "expected_return_today": profile.get(today, 0),
        }
