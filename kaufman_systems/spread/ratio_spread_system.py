class RatioSpreadSystem:
    """
    Ratio trading between two correlated assets
    """

    def __init__(self, lookback=100):
        self.lookback = lookback

    def signal(self, price_a, price_b):

        ratio = price_a / price_b
        mean = ratio.rolling(self.lookback).mean()

        if ratio.iloc[-1] > mean.iloc[-1]:
            return -1
        else:
            return 1

    def position_sizing(self, capital):

        return capital * 0.5

    def risk_filter(self, volatility):

        return volatility < 0.04
