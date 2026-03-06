class OpinionIndicatorSystem:

    def __init__(self, extreme=0.8):
        self.extreme = extreme

    def signal(self, opinion_index):

        if opinion_index.iloc[-1] > self.extreme:
            return -1

        if opinion_index.iloc[-1] < 1 - self.extreme:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.01 / risk

    def risk_filter(self, drawdown):
        return drawdown < 0.1
