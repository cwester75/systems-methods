class TermStructureSpread:

    def signal(self, near_contract, far_contract):

        spread = far_contract - near_contract

        if spread > 0:
            return 1
        else:
            return -1

    def position_sizing(self, capital):

        return capital * 0.4
