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

import numpy as np

from kaufman_systems.base import TradingSystem


class WeekdayPatternSystem(TradingSystem):

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

    def _weekday_returns(self, closes, weekdays):
        """Compute mean return per weekday over the lookback window."""
        closes = np.asarray(closes, dtype=float)
        weekdays = np.asarray(weekdays, dtype=int)

        returns = np.diff(closes) / closes[:-1]
        wdays = weekdays[1:]

        n = min(self.lookback, len(returns))
        returns = returns[-n:]
        wdays = wdays[-n:]

        means = {}
        for d in range(5):
            mask = wdays == d
            if np.any(mask):
                means[d] = float(np.mean(returns[mask]))
            else:
                means[d] = 0.0
        return means

    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``closes``, ``weekdays`` (array of int 0-4),
        and ``weekday`` (int, current day).

        Returns
        -------
        1  -> long
        -1 -> short
        0  -> neutral
        """
        weekday_profile = self._weekday_returns(data["closes"], data["weekdays"])
        today = data["weekday"]

        expected_return = weekday_profile.get(today, 0)

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
        profile = self._weekday_returns(data["closes"], data["weekdays"])
        return {"weekday_returns": profile}
