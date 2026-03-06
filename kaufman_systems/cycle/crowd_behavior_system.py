class CrowdBehaviorSystem:

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def signal(self, crowd_index):

        if crowd_index.iloc[-1] > self.threshold:
            return -1

        if crowd_index.iloc[-1] < 1 - self.threshold:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.015 / risk

    def risk_filter(self, drawdown):
        return drawdown < 0.12
