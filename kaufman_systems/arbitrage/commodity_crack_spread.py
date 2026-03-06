class CommodityCrackSpread:

    """
    Oil vs refined products spread
    """

    def signal(self, crude, gasoline, heating_oil):

        crack = gasoline + heating_oil - crude

        if crack > 0:
            return 1
        else:
            return -1

    def position_sizing(self, capital):

        return capital * 0.4
