class CarrySpreadSystem:
    """
    Carry differential system
    Example: FX carry
    """

    def signal(self, yield_a, yield_b):

        carry = yield_a - yield_b

        if carry > 0:
            return 1
        else:
            return -1

    def position_sizing(self, capital):

        return capital * 0.3
