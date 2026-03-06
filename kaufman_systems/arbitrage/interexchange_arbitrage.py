class InterExchangeArbitrage:

    def signal(self, price_a, price_b, threshold=0.002):

        diff = (price_a - price_b) / price_b

        if diff > threshold:
            return -1
        elif diff < -threshold:
            return 1

        return 0

    def position_sizing(self, capital):

        return capital * 0.5
