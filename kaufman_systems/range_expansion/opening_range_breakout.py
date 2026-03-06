# range_expansion/opening_range_breakout.py

import numpy as np
import pandas as pd


class OpeningRangeBreakout:
    """
    Opening Range Breakout (ORB) System

    Concept
    -------
    The opening range represents the price discovery period
    immediately after market open. Breakouts from this range
    often produce directional intraday trends.

    Opening Range
    -------------
    Defined as the high and low of the first N bars of the session.

    Trading Logic
    -------------

    Long Entry:
        price breaks above opening range high

    Short Entry:
        price breaks below opening range low

    Exit:
        end of session or ATR stop
    """

    def __init__(
        self,
        opening_bars: int = 5,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):

        self.opening_bars = opening_bars
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
    # Opening Range
    # -------------------------------------------------

    def compute_opening_range(self, data: pd.DataFrame):

        """
        Assumes intraday data with a DateTime index.
        """

        df = data.copy()
        df["date"] = df.index.date

        opening_high = []
        opening_low = []

        for d, group in df.groupby("date"):

            first = group.iloc[: self.opening_bars]

            high = first["high"].max()
            low = first["low"].min()

            opening_high.extend([high] * len(group))
            opening_low.extend([low] * len(group))

        df["opening_high"] = opening_high
        df["opening_low"] = opening_low

        return df["opening_high"], df["opening_low"]

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        close = data["close"]

        opening_high, opening_low = self.compute_opening_range(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = close > opening_high
        short_entry = close < opening_low

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

        position_size = risk_dollars / (atr * self.atr_multiplier)

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
