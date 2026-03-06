"""
Intraday Range Pattern System
-----------------------------

Concept
-------
The early session often establishes a directional range.
Breakouts beyond this range frequently lead to trend moves.

Rules
-----
LONG
    price breaks above opening range high

SHORT
    price breaks below opening range low

Exit
    end of session or risk stop
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class IntradayRangePatternSystem(TradingSystem):

    def __init__(
        self,
        range_minutes: int = 30,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        range_minutes : int
            time window used to define opening range
        risk_per_trade : float
            portfolio risk fraction
        """
        self.range_minutes = range_minutes
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``closes`` and ``opening_range``
        (a tuple/list of ``(range_high, range_low)``).

        Returns
        -------
        1  -> long breakout
        -1 -> short breakout
        0  -> no signal
        """
        closes = np.asarray(data["closes"], dtype=float)
        range_high, range_low = data["opening_range"]

        price = closes[-1]

        if price > range_high:
            return 1

        if price < range_low:
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

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def indicators(self, data: dict) -> dict:
        range_high, range_low = data["opening_range"]
        return {
            "range_high": range_high,
            "range_low": range_low,
        }
