class MarketSentimentIndexSystem:

    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def signal(self, sentiment_index):

        if sentiment_index.iloc[-1] > self.threshold:
            return -1

        if sentiment_index.iloc[-1] < 1 - self.threshold:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.02 / risk

    def risk_filter(self, volume):
        return volume > volume.mean()
