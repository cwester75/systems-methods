class MeanReversionSpread:

    def __init__(self, lookback=50):

        self.lookback = lookback

    def signal(self, spread):

        mean = spread.rolling(self.lookback).mean()

        if spread.iloc[-1] > mean.iloc[-1]:
            return -1
        else:
            return 1

    def position_sizing(self, capital):

        return capital * 0.4
