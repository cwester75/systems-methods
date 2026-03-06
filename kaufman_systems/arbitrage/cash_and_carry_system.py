class CashAndCarrySystem:

    def signal(self, spot_price, futures_price, carry_cost):

        fair = spot_price + carry_cost

        if futures_price > fair:
            return -1
        elif futures_price < fair:
            return 1

        return 0

    def position_sizing(self, capital):

        return capital * 0.5
