"""
Smoke tests for the Phase 4 spread / arbitrage systems.

Each system gets crafted price data that really triggers LONG / SHORT,
plus indicator-key, insufficient-data, and cross-cutting sizing checks.
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.spread.intermarket_spread_system import IntermarketSpreadSystem
from kaufman_systems.spread.ratio_spread_system import RatioSpreadSystem
from kaufman_systems.spread.pairs_trading_system import PairsTradingSystem
from kaufman_systems.spread.mean_reversion_spread import MeanReversionSpreadSystem
from kaufman_systems.spread.carry_spread_system import CarrySpreadSystem
from kaufman_systems.spread.term_structure_spread import TermStructureSpreadSystem
from kaufman_systems.arbitrage.cash_and_carry_system import CashAndCarrySystem
from kaufman_systems.arbitrage.calendar_spread_system import CalendarSpreadSystem
from kaufman_systems.arbitrage.commodity_crack_spread import CommodityCrackSpreadSystem
from kaufman_systems.arbitrage.interexchange_arbitrage import InterexchangeArbitrageSystem


# ===================================================================
# Helpers
# ===================================================================

def _make_data(closes, spread=1.0):
    closes = np.asarray(closes, dtype=float)
    return {
        "closes": closes,
        "highs": closes + spread,
        "lows": closes - spread,
    }


def _flat_with_last(last, n=100, base=100.0):
    closes = np.full(n, base)
    closes[-1] = last
    return _make_data(closes)


RISK = {"equity": 100_000.0}

DATA_UP = _make_data(np.linspace(100, 200, 100))
DATA_DOWN = _make_data(np.linspace(200, 100, 100))
DATA_FLAT = _make_data(np.full(100, 150.0))
DATA_SHORT = _make_data([100.0, 101.0])

# Small deterministic oscillation so AR(1) regression is well conditioned
_osc = 100.0 + 0.5 * np.sin(np.arange(100))


# ===================================================================
# Spread systems
# ===================================================================

class TestIntermarketSpreadSystem:

    def setup_method(self):
        self.sys = IntermarketSpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_far_below_mean(self):
        assert self.sys.signal(_flat_with_last(90.0)) == 1

    def test_short_when_far_above_mean(self):
        assert self.sys.signal(_flat_with_last(110.0)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_flat_on_steady_trend(self):
        # Last bar of a linear ramp sits ~1.7 sd from the mean, under entry_z
        assert self.sys.signal(DATA_UP) == 0

    def test_indicators_keys(self):
        assert "zscore" in self.sys.indicators(DATA_UP)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestRatioSpreadSystem:

    def setup_method(self):
        self.sys = RatioSpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["ratio"] > 1.02

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestPairsTradingSystem:

    def setup_method(self):
        self.sys = PairsTradingSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_large_negative_residual(self):
        closes = _osc.copy()
        closes[-1] = 90.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_short_on_large_positive_residual(self):
        closes = _osc.copy()
        closes[-1] = 110.0
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_when_no_outlier(self):
        assert self.sys.signal(_make_data(_osc)) == 0

    def test_indicators_keys(self):
        assert "residual_zscore" in self.sys.indicators(_make_data(_osc))

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestMeanReversionSpreadSystem:

    def setup_method(self):
        self.sys = MeanReversionSpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_far_below_ma(self):
        assert self.sys.signal(_flat_with_last(90.0)) == 1

    def test_short_when_far_above_ma(self):
        assert self.sys.signal(_flat_with_last(110.0)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(_flat_with_last(90.0))
        assert ind["atr_deviation"] < -2.0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestCarrySpreadSystem:

    def setup_method(self):
        self.sys = CarrySpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_positive_slope(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_negative_slope(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_zero_slope(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["normalized_slope"] > 0.001

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestTermStructureSpreadSystem:

    def setup_method(self):
        self.sys = TermStructureSpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_fast_momentum_exceeds_slow(self):
        # Older bars high (120), then drop to 100 and bounce to 105:
        # fast_mom = +5%, slow_mom = -12.5%
        closes = np.full(100, 100.0)
        closes[:50] = 120.0
        closes[-1] = 105.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_short_when_slow_momentum_exceeds_fast(self):
        # Older bars low (80), recent flat at 100: fast_mom = 0, slow_mom = +25%
        closes = np.full(100, 100.0)
        closes[:50] = 80.0
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        assert "momentum_spread" in self.sys.indicators(DATA_UP)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# Arbitrage systems
# ===================================================================

class TestCashAndCarrySystem:

    def setup_method(self):
        self.sys = CashAndCarrySystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_below_fair_value(self):
        assert self.sys.signal(_flat_with_last(90.0)) == 1

    def test_short_when_above_fair_value(self):
        assert self.sys.signal(_flat_with_last(110.0)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "fair_value" in ind
        assert "atr_deviation" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestCalendarSpreadSystem:

    def setup_method(self):
        self.sys = CalendarSpreadSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_near_return_beats_far_return(self):
        # far_ref = 110, near_ref = 100, price = 100 -> near 0%, far -9%
        closes = np.full(100, 100.0)
        closes[:80] = 110.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_short_when_far_return_beats_near_return(self):
        # far_ref = 90, near_ref = 100, price = 100 -> near 0%, far +11%
        closes = np.full(100, 100.0)
        closes[:80] = 90.0
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        assert "calendar_spread" in self.sys.indicators(DATA_UP)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestCommodityCrackSpreadSystem:

    def setup_method(self):
        self.sys = CommodityCrackSpreadSystem()

    def _data_with_last_half_range(self, half_range):
        closes = np.full(100, 100.0)
        hw = np.full(100, 1.0)
        hw[-1] = half_range
        return {"closes": closes, "highs": closes + hw, "lows": closes - hw}

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_range_expansion(self):
        assert self.sys.signal(self._data_with_last_half_range(5.0)) == 1

    def test_short_on_range_compression(self):
        assert self.sys.signal(self._data_with_last_half_range(0.1)) == -1

    def test_flat_on_constant_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "crack" in ind
        assert "upper_threshold" in ind
        assert "lower_threshold" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestInterexchangeArbitrageSystem:

    def setup_method(self):
        self.sys = InterexchangeArbitrageSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_below_channel_midpoint(self):
        assert self.sys.signal(_flat_with_last(90.0)) == 1

    def test_short_when_above_channel_midpoint(self):
        assert self.sys.signal(_flat_with_last(110.0)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "midpoint" in ind
        assert "atr_deviation" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# Cross-cutting
# ===================================================================

class TestPhase4Sizing:

    ALL = [
        IntermarketSpreadSystem(),
        RatioSpreadSystem(),
        PairsTradingSystem(),
        MeanReversionSpreadSystem(),
        CarrySpreadSystem(),
        TermStructureSpreadSystem(),
        CashAndCarrySystem(),
        CalendarSpreadSystem(),
        CommodityCrackSpreadSystem(),
        InterexchangeArbitrageSystem(),
    ]

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_scales_with_equity(self, sys):
        size_100k = sys.position_sizing(DATA_UP, {"equity": 100_000.0})
        size_200k = sys.position_sizing(DATA_UP, {"equity": 200_000.0})
        assert size_100k > 0
        assert abs(size_200k / size_100k - 2.0) < 0.01

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_zero_on_insufficient_data(self, sys):
        assert sys.position_sizing(DATA_SHORT, RISK) == 0

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_insufficient_data(self, sys):
        assert sys.risk_filter(DATA_SHORT) is False

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_zero_vol(self, sys):
        zero_vol = {
            "closes": np.full(100, 100.0),
            "highs": np.full(100, 100.0),
            "lows": np.full(100, 100.0),
        }
        assert sys.risk_filter(zero_vol) is False
