class GannTimeCycleSystem:

    def __init__(self, cycle=90):
        self.cycle = cycle

    def signal(self, day):

        if day % self.cycle == 0:
            return 1

        if day % (self.cycle / 2) == 0:
            return -1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.015 / risk

    def risk_filter(self, volatility):
        return volatility < 0.04
