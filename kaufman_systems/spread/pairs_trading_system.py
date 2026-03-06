import statsmodels.api as sm

class PairsTradingSystem:
    """
    Cointegration-based pairs trading
    """

    def __init__(self, lookback=120, entry=2, exit=0.5):
        self.lookback = lookback
        self.entry = entry
        self.exit = exit

    def hedge_ratio(self, x, y):

        model = sm.OLS(y, sm.add_constant(x)).fit()
        return model.params[1]

    def signal(self, x, y):

        hr = self.hedge_ratio(x[-self.lookback:], y[-self.lookback:])

        spread = y - hr * x

        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()

        z = (spread - mean) / std

        if z.iloc[-1] > self.entry:
            return -1
        elif z.iloc[-1] < -self.entry:
            return 1
        elif abs(z.iloc[-1]) < self.exit:
            return 0

        return 0

    def position_sizing(self, capital):

        return capital * 0.5
