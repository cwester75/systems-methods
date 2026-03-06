import pandas as pd


class COTPositioningSystem:

    def __init__(self, lookback=156):
        self.lookback = lookback

    def signal(self, cot_df):

        net = cot_df["commercial_long"] - cot_df["commercial_short"]

        max_extreme = net.rolling(self.lookback).max()
        min_extreme = net.rolling(self.lookback).min()

        if net.iloc[-1] >= max_extreme.iloc[-1]:
            return 1

        if net.iloc[-1] <= min_extreme.iloc[-1]:
            return -1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.02 / risk

    def risk_filter(self, volatility):
        return volatility < 0.04
