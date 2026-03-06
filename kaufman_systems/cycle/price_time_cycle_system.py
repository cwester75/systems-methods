import numpy as np


class PriceTimeCycleSystem:

    def __init__(self, cycle=30):
        self.cycle = cycle

    def signal(self, price_series):

        t = len(price_series)

        phase = np.sin(2 * np.pi * t / self.cycle)

        if phase > 0.8:
            return -1

        if phase < -0.8:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.015 / risk

    def risk_filter(self, volatility):
        return volatility < 0.05
