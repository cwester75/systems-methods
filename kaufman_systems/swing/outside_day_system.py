# swing/outside_day_system.py

import numpy as np
import pandas as pd


class OutsideDaySystem:
    """
    Outside Day Reversal System

    Concept
    -------
    An Outside Day occurs when:

        today's high > yesterday's high
        AND
        today's low < yesterday's low

    This indicates a volatility expansion and possible reversal.

    Trading Logic
    -------------

    Bullish Outside Day:
        outside day AND close > open
        -> go LONG next bar

    Bearish Outside Day:
        outside day AND close < open
        -> go SHORT next bar

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
    # Signal Detection
    # -------------------------------------------------

    def detect_outside_days(self, data: pd.DataFrame):

        prev_high = data["high"].shift(1)
        prev_low = data["low"].shift(1)

        outside_day = (data["high"] > prev_high) & (data["low"] < prev_low)

        bullish = outside_day & (data["close"] > data["open"])
        bearish = outside_day & (data["close"] < data["open"])

        return bullish, bearish

    # -------------------------------------------------
    # Signal Generation
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        bullish, bearish = self.detect_outside_days(data)

        signal = pd.Series(index=data.index, dtype=float)

        signal[bullish] = 1
        signal[bearish] = -1

        signal = signal.shift(1)  # trade next bar
        signal = signal.ffill().fillna(0)

        # enforce maximum holding period
        holding = 0
        for i in range(len(signal)):
            if signal.iloc[i] != 0:
                holding += 1
                if holding > self.max_hold:
                    signal.iloc[i] = 0
                    holding = 0
            else:
                holding = 0

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

        mask = vol_ratio > 0.003

        return mask

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
