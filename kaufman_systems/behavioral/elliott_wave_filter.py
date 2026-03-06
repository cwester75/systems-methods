import numpy as np


class ElliottWaveFilter:

    def signal(self, price):

        diffs = np.diff(price)

        if len(diffs) < 5:
            return 0

        waves = np.sign(diffs[-5:])

        if list(waves) == [1, -1, 1, -1, 1]:
            return 1

        if list(waves) == [-1, 1, -1, 1, -1]:
            return -1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.02 / risk

    def risk_filter(self, trend_strength):
        return trend_strength > 0.5
