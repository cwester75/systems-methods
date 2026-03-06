# volatility_contraction/volatility_ratio_system.py

import numpy as np
import pandas as pd


class VolatilityRatioSystem:
    """
    Volatility Ratio Expansion System

    Concept
    -------
    Detect volatility regime shifts by comparing short-term
    volatility to long-term volatility.

    Volatility Ratio:
        VR = short_vol / long_vol

    When volatility expands rapidly (VR >> 1), markets often
    transition into trending regimes.

    Trading Logic
    -------------
    Long Entry:
        volatility_ratio > upper_threshold AND price momentum positive

    Short Entry:
        volatility_ratio > upper_threshold AND price momentum negative

    Exit:
        volatility_ratio falls back below exit_threshold
    """

    def __init__(
        self,
        short_vol: int = 10,
        long_vol: int = 50,
        vr_threshold: float = 1.5,
        exit_threshold: float = 1.0,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):
        self.short_vol = short_vol
        self.long_vol = long_vol
        self.vr_threshold = vr_threshold
        self.exit_threshold = exit_threshold
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.risk_per_trade = risk_per_trade

    # -------------------------------------------------
    # Volatility Measures
    # -------------------------------------------------

    def volatility(self, series: pd.Series, window: int):

        returns = series.pct_change()

        return returns.rolling(window).std()

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
    # Volatility Ratio
    # -------------------------------------------------

    def volatility_ratio(self, data: pd.DataFrame):

        close = data["close"]

        short_vol = self.volatility(close, self.short_vol)
        long_vol = self.volatility(close, self.long_vol)

        vr = short_vol / long_vol

        return vr

    # -------------------------------------------------
    # Momentum Filter
    # -------------------------------------------------

    def momentum(self, data: pd.DataFrame, lookback: int = 5):

        return data["close"] - data["close"].shift(lookback)

    # -------------------------------------------------
    # Signal Logic
    # -------------------------------------------------

    def signal(self, data: pd.DataFrame):

        vr = self.volatility_ratio(data)
        mom = self.momentum(data)

        signal = pd.Series(index=data.index, dtype=float)

        long_entry = (vr > self.vr_threshold) & (mom > 0)
        short_entry = (vr > self.vr_threshold) & (mom < 0)

        signal[long_entry] = 1
        signal[short_entry] = -1

        signal = signal.ffill().fillna(0)

        # Exit when volatility normalizes
        exit_condition = vr < self.exit_threshold
        signal[exit_condition] = 0

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
