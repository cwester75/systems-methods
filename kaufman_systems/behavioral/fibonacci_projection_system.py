class FibonacciProjectionSystem:

    def signal(self, price_series):

        swing_low = price_series.min()
        swing_high = price_series.max()

        fib_target = swing_high + (swing_high - swing_low) * 0.618

        if price_series.iloc[-1] >= fib_target:
            return -1

        return 1

    def position_sizing(self, capital, risk):
        return capital * 0.02 / risk

    def risk_filter(self, volatility):
        return volatility < 0.06
