"""
Volatility Expansion Breakout System

Concept
-------
Detects when the current bar's range expands significantly compared to
a short-term average range, AND the close is near the bar's extreme.
This combines range expansion with directional conviction (close near
high = bullish, close near low = bearish).

Signal logic
------------
If current range > expansion_mult × avg_range(lookback):
  If close in upper 25% of bar → LONG
  If close in lower 25% of bar → SHORT
Otherwise                      → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class VolatilityExpansionBreakoutSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 10,
        expansion_mult: float = 1.5,
        close_pct: float = 0.25,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.expansion_mult = expansion_mult
        self.close_pct = close_pct
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

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(highs) < self.lookback + 1:
            return 0

        ranges = highs - lows
        avg_range = np.mean(ranges[-(self.lookback + 1):-1])
        current_range = ranges[-1]

        if avg_range == 0:
            return 0

        if current_range <= self.expansion_mult * avg_range:
            return 0

        if current_range == 0:
            return 0

        close_position = (closes[-1] - lows[-1]) / current_range

        if close_position >= (1 - self.close_pct):
            return 1

        if close_position <= self.close_pct:
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
        highs = np.asarray(highs)
        lows = np.asarray(lows)

        if len(highs) < self.lookback + 1:
            return {"current_range": None, "avg_range": None, "expansion_ratio": None}

        ranges = highs - lows
        avg_range = np.mean(ranges[-(self.lookback + 1):-1])
        current_range = ranges[-1]

        return {
            "current_range": current_range,
            "avg_range": avg_range,
            "expansion_ratio": current_range / avg_range if avg_range != 0 else None,
        }
# range_expansion/volatility_expansion_breakout.py

import numpy as np
import pandas as pd


class VolatilityExpansionBreakout:
    """
    Volatility Expansion Breakout System

    Concept
    -------
    Markets alternate between low and high volatility regimes.
    A volatility expansion event (ATR rising rapidly relative
    to its recent history) often precedes directional movement.

    Volatility Expansion
    --------------------
    ATR_ratio = ATR_short / ATR_long

    Expansion occurs when:
        ATR_ratio > expansion_threshold

    Trading Logic
    -------------
    Long Entry:
        volatility expansion AND price > recent high

    Short Entry:
        volatility expansion AND price < recent low

    Exit:
        price crosses midpoint or volatility normalizes
    """

    def __init__(
        self,
        atr_short: int = 10,
        atr_long: int = 40,
        breakout_length: int = 20,
        expansion_threshold: float = 1.4,
        atr_stop_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.atr_short = atr_short
        self.atr_long = atr_long
        self.breakout_length = breakout_length
        self.expansion_threshold = expansion_threshold
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # ATR Calculation
    # -------------------------------------------------

    def atr(self, data: pd.DataFrame, window: int):

        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return tr.rolling(window).mean()

    # -------------------------------------------------
    # Volatility Expansion
    # -------------------------------------------------

    def volatility_ratio(self, data: pd.DataFrame):

        atr_short = self.atr(data, self.atr_short)
        atr_long = self.atr(data, self.atr_long)

        return atr_short / atr_long

    # -------------------------------------------------
    # Breakout Levels
    # -------------------------------------------------

    def breakout_levels(self, data: pd.DataFrame):

        upper = data["high"].rolling(self.breakout_length).max()
        lower = data["low"].rolling(self.breakout_length).min()

        return upper, lower

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        vr = self.volatility_ratio(data)
        upper, lower = self.breakout_levels(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (vr > self.expansion_threshold) & (close > upper.shift(1))
        short_entry = (vr > self.expansion_threshold) & (close < lower.shift(1))

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit when volatility contracts
        exit_condition = vr < 1.0
        signal[exit_condition] = 0

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data, self.atr_short)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_stop_multiplier)

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data, self.atr_short)

        vol_ratio = atr / data["close"]

        return vol_ratio > 0.002

    # -------------------------------------------------
    # Backtest Runner
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
