import numpy as np
import pandas as pd

class IntermarketSpreadSystem:
    """
    Intermarket spread between two related assets
    Example: SPY vs TLT or Crude vs Energy equities
    """

    def __init__(self, lookback=60, entry_z=1.5, exit_z=0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def signal(self, price_a: pd.Series, price_b: pd.Series):

        spread = price_a - price_b
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()

        z = (spread - mean) / std

        if z.iloc[-1] > self.entry_z:
            return -1
        elif z.iloc[-1] < -self.entry_z:
            return 1
        elif abs(z.iloc[-1]) < self.exit_z:
            return 0

        return 0

    def position_sizing(self, capital):

        return capital * 0.5

    def risk_filter(self, volatility):

        return volatility < 0.03
