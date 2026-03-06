# pattern/inside_day_breakout.py

import numpy as np
import pandas as pd


class InsideDayBreakoutSystem:
    """
    Inside Day Breakout System

    Concept
    -------
    An inside day occurs when:

        today's high < yesterday's high
        AND
        today's low > yesterday's low

    This indicates volatility compression. A breakout of the
    inside-day range often leads to directional movement.

    Trading Logic
    -------------
    Long:
        price breaks above inside-day high

    Short:
        price breaks below inside-day low

    Exit:
        opposite signal or ATR stop
    """

    def __init__(
        self,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        max_hold: int = 10,
    ):
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
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
    # Inside Day Detection
    # -------------------------------------------------

    def detect_inside_day(self, data: pd.DataFrame):

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        inside_day = (data["high"] < prev_high) & (data["low"] > prev_low)

        inside_high = data["high"]
        inside_low = data["low"]

        return inside_day, inside_high, inside_low

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        inside_day, inside_high, inside_low = self.detect_inside_day(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_breakout = (close > inside_high.shift(1)) & inside_day.shift(1)
        short_breakout = (close < inside_low.shift(1)) & inside_day.shift(1)

        signal[long_breakout] = 1
        signal[short_breakout] = -1

        signal = signal.ffill().fillna(0)

        # enforce max holding period
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
