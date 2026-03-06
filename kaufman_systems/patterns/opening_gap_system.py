"""
Opening Gap Pattern System
--------------------------

Concept
-------
Large overnight gaps often lead to continuation moves during the session.

Rules
-----
LONG
    gap_up > gap_threshold * ATR

SHORT
    gap_down > gap_threshold * ATR

Exit
    end-of-day or opposite signal
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class OpeningGapSystem(TradingSystem):

    def __init__(
        self,
        gap_threshold: float = 1.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        gap_threshold : float
            gap size relative to ATR required to trigger signal
        atr_period : int
            ATR lookback
        risk_per_trade : float
            portfolio risk per trade
        """
        self.gap_threshold = gap_threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # ATR Calculation
    # ---------------------------------------------------------

    def _atr(self, highs, lows, closes):
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)

        if len(closes) < self.atr_period + 1:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        return np.mean(tr[-self.atr_period:])

    # ---------------------------------------------------------
    # Signal Generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        ``data`` must include ``closes``, ``highs``, ``lows``, and ``opens``.

        Returns
        -------
        1 = long
        -1 = short
        0 = no signal
        """
        closes = np.asarray(data["closes"], dtype=float)
        highs = np.asarray(data["highs"], dtype=float)
        lows = np.asarray(data["lows"], dtype=float)
        opens = np.asarray(data["opens"], dtype=float)

        atr = self._atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        gap = opens[-1] - closes[-2]

        if gap > self.gap_threshold * atr:
            return 1

        if gap < -self.gap_threshold * atr:
            return -1

        return 0

    # ---------------------------------------------------------
    # Position Sizing
    # ---------------------------------------------------------

    def position_sizing(self, data: dict, risk: dict) -> float:
        """ATR-based position sizing."""
        highs = np.asarray(data["highs"], dtype=float)
        lows = np.asarray(data["lows"], dtype=float)
        closes = np.asarray(data["closes"], dtype=float)

        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self._atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

    # ---------------------------------------------------------
    # Risk Filter
    # ---------------------------------------------------------

    def risk_filter(self, data: dict) -> bool:
        """Liquidity filter — requires volume above 20-day average."""
        volumes = np.asarray(data["volumes"], dtype=float)

        if len(volumes) < 20:
            return False

        avg_volume = np.mean(volumes[-20:])
        return float(volumes[-1]) > avg_volume

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def indicators(self, data: dict) -> dict:
        highs = np.asarray(data["highs"], dtype=float)
        lows = np.asarray(data["lows"], dtype=float)
        closes = np.asarray(data["closes"], dtype=float)
        opens = np.asarray(data["opens"], dtype=float)

        atr = self._atr(highs, lows, closes)
        gap = opens[-1] - closes[-2] if len(closes) >= 2 else 0

        return {"atr": atr, "gap": gap}
