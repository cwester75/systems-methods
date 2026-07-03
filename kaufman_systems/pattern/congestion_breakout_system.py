"""
Congestion Breakout System

Concept
-------
Detects a congestion zone (tight consolidation) and signals when price
breaks out.  Congestion is defined as N consecutive bars where the
range (high-low) stays below a fraction of ATR.

Signal logic
------------
If the last N bars form a congestion zone:
  Close > max(highs of congestion) → LONG
  Close < min(lows of congestion)  → SHORT
Otherwise                          → FLAT

Kaufman describes congestion patterns as compressed-spring setups
where the subsequent breakout tends to be directional.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class CongestionBreakoutSystem(TradingSystem):

    def __init__(
        self,
        congestion_bars: int = 5,
        range_factor: float = 0.5,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.congestion_bars = congestion_bars
        self.range_factor = range_factor
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def atr(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

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

    def congestion_zone(self, highs, lows, closes):
        """Return (is_congestion, zone_high, zone_low)."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        needed = max(self.congestion_bars + 1, self.atr_period + 1)

        if len(closes) < needed + 1:
            return False, None, None

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return False, None, None

        zone_start = -(self.congestion_bars + 1)
        zone_end = -1
        zone_highs = highs[zone_start:zone_end]
        zone_lows = lows[zone_start:zone_end]
        zone_ranges = zone_highs - zone_lows

        if np.all(zone_ranges < self.range_factor * atr):
            return True, np.max(zone_highs), np.min(zone_lows)

        return False, None, None

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        is_congestion, zone_high, zone_low = self.congestion_zone(
            highs, lows, closes
        )

        if not is_congestion:
            return 0

        price = np.asarray(closes)[-1]

        if price > zone_high:
            return 1

        if price < zone_low:
            return -1

        return 0

    def position_sizing(self, data, risk):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None or atr <= 0:
            return False

        return True

    def indicators(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        is_congestion, zone_high, zone_low = self.congestion_zone(
            highs, lows, closes
        )

        return {
            "is_congestion": is_congestion,
            "zone_high": zone_high,
            "zone_low": zone_low,
        }
