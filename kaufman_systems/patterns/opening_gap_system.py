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

import pandas as pd
import numpy as np


class OpeningGapSystem:

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

    def atr(self, df: pd.DataFrame) -> pd.Series:

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            ],
            axis=1
        ).max(axis=1)

        atr = tr.rolling(self.atr_period).mean()

        return atr


    # ---------------------------------------------------------
    # Signal Generation
    # ---------------------------------------------------------

    def signal(self, df: pd.DataFrame) -> int:
        """
        Returns
        -------
        1 = long
        -1 = short
        0 = no signal
        """

        atr = self.atr(df).iloc[-1]

        open_price = df["open"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        gap = open_price - prev_close

        if gap > self.gap_threshold * atr:
            return 1

        elif gap < -self.gap_threshold * atr:
            return -1

        return 0


    # ---------------------------------------------------------
    # Position Sizing
    # ---------------------------------------------------------

    def position_sizing(
        self,
        capital: float,
        atr_value: float
    ) -> float:
        """
        ATR-based position sizing
        """

        risk_dollars = capital * self.risk_per_trade

        position = risk_dollars / atr_value

        return position


    # ---------------------------------------------------------
    # Risk Filter
    # ---------------------------------------------------------

    def risk_filter(self, df: pd.DataFrame) -> bool:
        """
        Optional liquidity filter
        """

        volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]

        return volume > avg_volume
