"""
ATR Contraction System

Concept
-------
Detects when ATR contracts to a low relative to its recent history,
signaling volatility compression.  When current ATR drops below a
fraction of its rolling average, a contraction is detected.  The
breakout direction from the prior close determines the signal.

Signal logic
------------
If ATR < contraction_pct × SMA(ATR, lookback):
  Close > prev close + current ATR → LONG
  Close < prev close - current ATR → SHORT
Otherwise                          → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ATRContractionSystem(TradingSystem):

    def __init__(
        self,
        atr_period: int = 14,
        lookback: int = 50,
        contraction_pct: float = 0.75,
        risk_per_trade: float = 0.01,
    ):
        self.atr_period = atr_period
        self.lookback = lookback
        self.contraction_pct = contraction_pct
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

    def atr_series(self, highs, lows, closes):
        """Compute rolling ATR for the last `lookback` bars."""
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        needed = self.atr_period + 1 + self.lookback

        if len(closes) < needed:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )

        atrs = []
        for i in range(self.lookback):
            end = len(tr) - i
            start = end - self.atr_period
            atrs.append(np.mean(tr[start:end]))

        return np.array(atrs[::-1])

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        closes = np.asarray(closes)

        atr_vals = self.atr_series(highs, lows, closes)

        if atr_vals is None:
            return 0

        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals)

        if avg_atr == 0:
            return 0

        if current_atr >= self.contraction_pct * avg_atr:
            return 0

        if len(closes) < 2:
            return 0

        prev_close = closes[-2]
        price = closes[-1]

        if price > prev_close + current_atr:
            return 1

        if price < prev_close - current_atr:
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

        atr_vals = self.atr_series(highs, lows, closes)

        if atr_vals is None:
            return {"current_atr": None, "avg_atr": None, "is_contraction": False}

        current_atr = atr_vals[-1]
        avg_atr = np.mean(atr_vals)

        return {
            "current_atr": current_atr,
            "avg_atr": avg_atr,
            "is_contraction": current_atr < self.contraction_pct * avg_atr,
        }
# volatility_contraction/atr_contraction_system.py

import numpy as np
import pandas as pd


class ATRContractionSystem:
    """
    ATR Contraction Breakout System

    Concept
    -------
    Volatility contraction frequently precedes large directional moves.
    This system detects unusually low ATR relative to its historical
    distribution and trades the breakout.

    Contraction Condition
    ---------------------
        ATR < rolling percentile threshold

    Breakout
    --------
    Long:
        close > highest high of contraction window

    Short:
        close < lowest low of contraction window
    """

    def __init__(
        self,
        atr_length: int = 14,
        contraction_lookback: int = 100,
        contraction_percentile: float = 0.2,
        breakout_length: int = 20,
        atr_stop_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.atr_length = atr_length
        self.contraction_lookback = contraction_lookback
        self.contraction_percentile = contraction_percentile
        self.breakout_length = breakout_length
        self.atr_stop_multiplier = atr_stop_multiplier
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
    # Contraction Detection
    # -------------------------------------------------

    def detect_contraction(self, atr: pd.Series):

        threshold = atr.rolling(self.contraction_lookback).quantile(
            self.contraction_percentile
        )

        contraction = atr < threshold

        return contraction

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

        atr = self.atr(data)
        contraction = self.detect_contraction(atr)

        upper, lower = self.breakout_levels(data)

        close = data["close"]

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (close > upper.shift(1)) & contraction.shift(1)
        short_entry = (close < lower.shift(1)) & contraction.shift(1)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, data: pd.DataFrame, capital: float):

        atr = self.atr(data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_stop_multiplier)

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, data: pd.DataFrame):

        atr = self.atr(data)

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
