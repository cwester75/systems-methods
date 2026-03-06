# range_expansion/vix_expansion_system.py

import numpy as np
import pandas as pd


class VIXExpansionSystem:
    """
    VIX Expansion System

    Concept
    -------
    Volatility regime shifts often precede major market moves.
    This system monitors changes in the volatility index (VIX)
    relative to its recent trend.

    When VIX expands rapidly, risk regimes change and markets
    often transition into trending phases.

    Trading Logic
    -------------

    Long Risk Assets (e.g., SPY):
        VIX falling below its moving average

    Short Risk Assets:
        VIX spikes above expansion threshold

    Neutral:
        VIX normalizing

    Notes
    -----
    Requires two data inputs:

        price_data : asset being traded (SPY, futures, etc)
        vix_data   : VIX time series
    """

    def __init__(
        self,
        vix_ma_length: int = 20,
        vix_expansion_threshold: float = 1.25,
        atr_length: int = 14,
        atr_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
    ):

        self.vix_ma_length = vix_ma_length
        self.vix_expansion_threshold = vix_expansion_threshold
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
    # VIX Signals
    # -------------------------------------------------

    def vix_signal(self, vix_data: pd.DataFrame):

        vix = vix_data["close"]

        vix_ma = vix.rolling(self.vix_ma_length).mean()

        vix_ratio = vix / vix_ma

        signal = pd.Series(index=vix.index, dtype=float)

        # Risk-off regime
        signal[vix_ratio > self.vix_expansion_threshold] = -1

        # Risk-on regime
        signal[vix_ratio < 1.0] = 1

        signal = signal.ffill().fillna(0)

        return signal

    # -------------------------------------------------
    # Strategy Signal
    # -------------------------------------------------

    def signal(self, price_data: pd.DataFrame, vix_data: pd.DataFrame):

        vix_sig = self.vix_signal(vix_data)

        signal = vix_sig.copy()

        return signal

    # -------------------------------------------------
    # Position Sizing
    # -------------------------------------------------

    def position_sizing(self, price_data: pd.DataFrame, capital: float):

        atr = self.atr(price_data)

        risk_dollars = capital * self.risk_per_trade

        position_size = risk_dollars / (atr * self.atr_multiplier)

        return position_size

    # -------------------------------------------------
    # Risk Filter
    # -------------------------------------------------

    def risk_filter(self, price_data: pd.DataFrame):

        atr = self.atr(price_data)

        vol_ratio = atr / price_data["close"]

        return vol_ratio > 0.002

    # -------------------------------------------------
    # Backtest Runner
    # -------------------------------------------------

    def run(self, price_data: pd.DataFrame, vix_data: pd.DataFrame, capital: float = 100000):

        sig = self.signal(price_data, vix_data)

        size = self.position_sizing(price_data, capital)

        risk_mask = self.risk_filter(price_data)

        position = sig * size
        position = position.where(risk_mask, 0)

        returns = price_data["close"].pct_change()

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
