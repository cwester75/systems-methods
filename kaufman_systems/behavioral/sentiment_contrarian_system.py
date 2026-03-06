class SentimentContrarianSystem:

    def __init__(self, bull_threshold=0.75, bear_threshold=0.25):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def signal(self, sentiment):

        if sentiment.iloc[-1] > self.bull_threshold:
            return -1

        if sentiment.iloc[-1] < self.bear_threshold:
            return 1

        return 0

    def position_sizing(self, capital, risk):
        return capital * 0.015 / risk

    def risk_filter(self, volatility):
        return volatility < 0.05
