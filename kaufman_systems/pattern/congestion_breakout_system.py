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
# pattern/congestion_breakout_system.py

import numpy as np
import pandas as pd


class CongestionBreakoutSystem:
    """
    Congestion Breakout System

    Concept
    -------
    Markets often pause in a narrow consolidation ("congestion") before
    making a directional move. This system identifies low-range periods
    and trades the breakout.

    Congestion Detection
    --------------------
    A congestion zone occurs when:

        rolling_range < congestion_threshold

    where:
        rolling_range = (highest_high - lowest_low) / price

    Breakout
    --------
    Long:
        price breaks above congestion high

    Short:
        price breaks below congestion low
    """

    def __init__(
        self,
        congestion_length: int = 10,
        congestion_threshold: float = 0.02,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.congestion_length = congestion_length
        self.congestion_threshold = congestion_threshold
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # ATR
    # -------------------------------------------------

    def atr(self, data: pd.DataFrame):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr.rolling(self.atr_length).mean()

    # -------------------------------------------------
    # Congestion Detection
    # -------------------------------------------------

    def congestion_zone(self, data: pd.DataFrame):

        highest = data["high"].rolling(self.congestion_length).max()
        lowest = data["low"].rolling(self.congestion_length).min()

        range_pct = (highest - lowest) / data["close"]

        congestion = range_pct < self.congestion_threshold

        return congestion, highest, lowest

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        congestion, highest, lowest = self.congestion_zone(data)

        signal = pd.Series(index=data.index, dtype=float)

        breakout_long = (close > highest.shift(1)) & congestion.shift(1)
        breakout_short = (close < lowest.shift(1)) & congestion.shift(1)

        signal[breakout_long] = 1
        signal[breakout_short] = -1

        signal = signal.ffill().fillna(0)

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_multiplier)

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

        vol_ratio = atr / data["close"]

        return vol_ratio > 0.003

    # -------------------------------------------------
    # Run Backtest
    # -------------------------------------------------

    def run(self, data: pd.DataFrame, capital: float = 100000):

        sig = self.signal(data)
        size = self.position_sizing(data, capital)
        risk_mask = self.risk_filter(data)

        position = sig * size
        position = position.where(risk_mask, 0)

        returns = data["close"].pct_change()

        strat_returns = position.shift(1) * returns

        equity = (1 + strat_returns.fillna(0)).cumprod() * capital

        results = pd.DataFrame(
            {
                "signal": sig,
                "position": position,
                "strategy_returns": strat_returns,
                "equity": equity,
            }
        )

        return results
