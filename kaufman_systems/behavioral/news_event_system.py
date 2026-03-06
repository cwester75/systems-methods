import numpy as np


class NewsEventSystem:

    def __init__(self, threshold=2.5):
        self.threshold = threshold

    def signal(self, returns):

        z = (returns - returns.mean()) / returns.std()

        if z.iloc[-1] > self.threshold:
            return -1

        if z.iloc[-1] < -self.threshold:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.01 / risk

    def risk_filter(self, volume):
        return volume > volume.mean()
