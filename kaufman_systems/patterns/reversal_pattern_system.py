"""
Reversal Pattern System
-----------------------

Concept
-------
Reversal bars indicate exhaustion of the current move.

Common patterns detected:
    - Key Reversal Day
    - Outside Reversal Bar

Bullish Key Reversal
    new low + strong close above prior close

Bearish Key Reversal
    new high + strong close below prior close
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ReversalPatternSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 20,
        risk_per_trade: float = 0.01
    ):
        """
        Parameters
        ----------
        lookback : int
            period used to determine local extremes
        risk_per_trade : float
            portfolio risk per trade
        """
        self.lookback = lookback
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # ATR helper
    # ---------------------------------------------------------

    def _atr(self, highs, lows, closes, period=14):
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)

        if len(closes) < period + 1:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-period:])

    # ---------------------------------------------------------
    # Detect bullish reversal
    # ---------------------------------------------------------

    def _bullish_reversal(self, highs, lows, closes):
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)

        if len(closes) < self.lookback + 1:
            return False

        lowest = np.min(lows[-(self.lookback + 1):-1])

        return bool(lows[-1] < lowest and closes[-1] > closes[-2])

    # ---------------------------------------------------------
    # Detect bearish reversal
    # ---------------------------------------------------------

    def _bearish_reversal(self, highs, lows, closes):
        highs = np.asarray(highs, dtype=float)
        closes = np.asarray(closes, dtype=float)

        if len(closes) < self.lookback + 1:
            return False

        highest = np.max(highs[-(self.lookback + 1):-1])

        return bool(highs[-1] > highest and closes[-1] < closes[-2])

    # ---------------------------------------------------------
    # Signal generation
    # ---------------------------------------------------------

    def signal(self, data: dict) -> int:
        """
        Returns
        -------
        1  -> long
        -1 -> short
        0  -> neutral
        """
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        if self._bullish_reversal(highs, lows, closes):
            return 1

        if self._bearish_reversal(highs, lows, closes):
            return -1

        return 0

    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(self, data: dict, risk: dict) -> float:
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self._atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

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
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        return {
            "bullish_reversal": self._bullish_reversal(highs, lows, closes),
            "bearish_reversal": self._bearish_reversal(highs, lows, closes),
        }
