"""
Thrust System

Concept
-------
A thrust is a strong directional move measured as the percentage
change over N bars.  When the move exceeds a threshold, a trend
signal fires in the direction of the thrust.

Signal logic
------------
ROC(N) >  threshold → LONG
ROC(N) < -threshold → SHORT
Otherwise           → FLAT

Kaufman describes thrust methods as confirmation that a market has
enough momentum to sustain a directional move.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class ThrustSystem(TradingSystem):

    def __init__(
        self,
        thrust_period: int = 5,
        threshold: float = 0.03,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.thrust_period = thrust_period
        self.threshold = threshold
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    def thrust(self, closes):
        closes = np.asarray(closes)

        if len(closes) < self.thrust_period + 1:
            return None

        prev = closes[-(self.thrust_period + 1)]

        if prev == 0:
            return None

        return (closes[-1] - prev) / prev

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
        closes = data["closes"]

        t = self.thrust(closes)

        if t is None:
            return 0

        if t > self.threshold:
            return 1

        if t < -self.threshold:
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
        closes = data["closes"]

        return {
            "thrust": self.thrust(closes),
            "threshold": self.threshold,
        }
# swing/thrust_system.py

import numpy as np
import pandas as pd


class ThrustSystem:
    """
    Thrust Trading System

    Concept
    -------
    A "thrust" occurs when price moves strongly in one direction
    with large range and strong close. This indicates momentum
    expansion and potential continuation.

    Thrust Detection
    ----------------
    A bullish thrust occurs when:

        (close - open) / (high - low) > thrust_ratio
        AND
        (high - low) > k * ATR

    A bearish thrust occurs when:

        (open - close) / (high - low) > thrust_ratio
        AND
        (high - low) > k * ATR

    Trading Logic
    -------------
    Bullish Thrust:
        go LONG next bar

    Bearish Thrust:
        go SHORT next bar

    Exit
    ----
    Opposite signal or trailing ATR stop
    """

    def __init__(
        self,
        atr_length: int = 14,
        thrust_ratio: float = 0.7,
        range_multiplier: float = 1.5,
        atr_stop_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        max_hold: int = 15,
    ):

        self.atr_length = atr_length
        self.thrust_ratio = thrust_ratio
        self.range_multiplier = range_multiplier
        self.atr_stop_multiplier = atr_stop_multiplier
        self.risk_per_trade = risk_per_trade
        self.max_hold = max_hold

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
    # Thrust Detection
    # -------------------------------------------------

    def detect_thrust(self, data: pd.DataFrame):

        atr = self.atr(data)

        body = data["close"] - data["open"]
        range_ = data["high"] - data["low"]

        strength = body / range_.replace(0, np.nan)

        bullish = (
            (strength > self.thrust_ratio)
            & (range_ > self.range_multiplier * atr)
        )

        bearish = (
            (strength < -self.thrust_ratio)
            & (range_ > self.range_multiplier * atr)
        )

        return bullish, bearish

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        bullish, bearish = self.detect_thrust(data)

        signal = pd.Series(index=data.index, dtype=float)

        signal[bullish] = 1
        signal[bearish] = -1

        signal = signal.shift(1)  # enter next bar
        signal = signal.ffill().fillna(0)

        # enforce max hold
        hold = 0
        for i in range(len(signal)):
            if signal.iloc[i] != 0:
                hold += 1
                if hold > self.max_hold:
                    signal.iloc[i] = 0
                    hold = 0
            else:
                hold = 0

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

        return vol_ratio > 0.003

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
