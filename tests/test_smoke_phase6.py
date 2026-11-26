"""
Smoke tests for Phase 6 systems (adaptive, multi-timeframe, advanced).

Signal logic is exercised with crafted price data; each system gets its
own class, plus a cross-cutting class for position sizing / risk filter.
KaufmanAdaptiveSystem is covered in test_smoke_systems.py.
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.adaptive.adaptive_trend_system import AdaptiveTrendSystem
from kaufman_systems.adaptive.adaptive_momentum_system import AdaptiveMomentumSystem
from kaufman_systems.adaptive.adaptive_channel_system import AdaptiveChannelSystem
from kaufman_systems.multi_timeframe.dual_timeframe_trend import DualTimeframeTrendSystem
from kaufman_systems.multi_timeframe.triple_screen_system import TripleScreenSystem
from kaufman_systems.multi_timeframe.multi_resolution_momentum import MultiResolutionMomentumSystem
from kaufman_systems.advanced.volatility_weighted_trend import VolatilityWeightedTrendSystem
from kaufman_systems.advanced.noise_filtered_trend import NoiseFilteredTrendSystem
from kaufman_systems.advanced.price_volume_distribution_system import PriceVolumeDistributionSystem


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


RISK = {"equity": 100_000.0}

DATA_UP = _make_data(np.linspace(100, 200, 100))
DATA_DOWN = _make_data(np.linspace(200, 100, 100))
DATA_FLAT = _make_data(np.full(100, 150.0))
DATA_SHORT = _make_data([100.0, 101.0])


# ===================================================================
# AdaptiveTrendSystem
# ===================================================================

class TestAdaptiveTrendSystem:

    def setup_method(self):
        self.sys = AdaptiveTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_flat_period_is_base(self):
        assert self.sys.adaptive_period(DATA_FLAT["closes"]) == self.sys.base_period

    def test_period_within_bounds(self):
        rng = np.random.default_rng(1)
        closes = 100 + np.cumsum(rng.normal(0, 5, 200))
        closes = np.abs(closes) + 1
        p = self.sys.adaptive_period(closes)
        assert self.sys.min_period <= p <= self.sys.max_period

    def test_higher_volatility_lengthens_period(self):
        calm = 100 + 0.1 * (np.arange(60) % 2)
        wild = 100 + 3.0 * (np.arange(60) % 2)
        assert self.sys.adaptive_period(wild) > self.sys.adaptive_period(calm)

    def test_indicators_keys(self):
        assert "adaptive_period" in self.sys.indicators(DATA_UP)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# AdaptiveMomentumSystem
# ===================================================================

class TestAdaptiveMomentumSystem:

    def setup_method(self):
        self.sys = AdaptiveMomentumSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_zero_vol_uses_base_period(self):
        assert self.sys.adaptive_period(DATA_FLAT["closes"]) == self.sys.base_period

    def test_risk_filter_true_with_vol(self):
        assert self.sys.risk_filter(DATA_UP) is True

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "adaptive_period" in ind
        assert ind["momentum"] > 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# AdaptiveChannelSystem
# ===================================================================

class TestAdaptiveChannelSystem:

    def setup_method(self):
        self.sys = AdaptiveChannelSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_upside_breakout(self):
        closes = np.full(100, 100.0)
        closes[-1] = 120.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_short_on_downside_breakout(self):
        closes = np.full(100, 100.0)
        closes[-1] = 80.0
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_inside_channel(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        for key in ("atr", "ma", "upper", "lower"):
            assert key in ind
        assert ind["upper"] > ind["ma"] > ind["lower"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# DualTimeframeTrendSystem
# ===================================================================

class TestDualTimeframeTrendSystem:

    def setup_method(self):
        self.sys = DualTimeframeTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_flat_on_timeframe_disagreement(self):
        # Slow trend up, but last close pulled below the fast MA.
        closes = np.linspace(100, 200, 100)
        closes[-1] = 185.0
        assert self.sys.signal(_make_data(closes)) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["slow_ma"] is not None
        assert ind["fast_ma"] > ind["slow_ma"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["slow_ma"] is None


# ===================================================================
# TripleScreenSystem
# ===================================================================

def _triple_screen_long_closes():
    # Long-term uptrend, a sharp pullback (RSI oversold), then a bounce
    # that closes at the 5-bar high.
    closes = list(np.linspace(100, 300, 85))
    closes += [closes[-1] - 2.0 * i for i in range(1, 14)]
    closes.append(closes[-1] + 8.0)
    return np.array(closes)


class TestTripleScreenSystem:

    def setup_method(self):
        self.sys = TripleScreenSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_oversold_pullback_in_uptrend(self):
        data = _make_data(_triple_screen_long_closes(), spread=0.0)
        assert self.sys.rsi(data["closes"]) < self.sys.rsi_oversold
        assert self.sys.signal(data) == 1

    def test_short_on_overbought_rally_in_downtrend(self):
        data = _make_data(400.0 - _triple_screen_long_closes(), spread=0.0)
        assert self.sys.rsi(data["closes"]) > self.sys.rsi_overbought
        assert self.sys.signal(data) == -1

    def test_flat_when_rsi_not_extreme(self):
        # Pure uptrend: RSI = 100, not oversold.
        assert self.sys.signal(DATA_UP) == 0
        assert self.sys.signal(DATA_DOWN) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "trend_ma" in ind
        assert ind["rsi"] == 100

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["rsi"] is None


# ===================================================================
# MultiResolutionMomentumSystem
# ===================================================================

class TestMultiResolutionMomentumSystem:

    def setup_method(self):
        self.sys = MultiResolutionMomentumSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_all_horizons_up(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_when_all_horizons_down(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_when_horizons_disagree(self):
        closes = np.linspace(100, 200, 100)
        closes[-3:] = [190.0, 180.0, 170.0]  # short horizon turns down
        assert self.sys.signal(_make_data(closes)) == 0

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["m_short"] > 0 and ind["m_medium"] > 0 and ind["m_long"] > 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["m_long"] is None


# ===================================================================
# VolatilityWeightedTrendSystem
# ===================================================================

class TestVolatilityWeightedTrendSystem:

    def setup_method(self):
        self.sys = VolatilityWeightedTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_smooth_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_smooth_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_choppy_no_trend(self):
        closes = np.where(np.arange(100) % 2 == 0, 100.0, 102.0)
        assert self.sys.signal(_make_data(closes)) == 0

    def test_flat_on_zero_vol(self):
        assert self.sys.signal(DATA_FLAT) == 0
        assert self.sys.risk_filter(DATA_FLAT) is False

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["trend_strength"] > self.sys.strength_threshold

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["trend_strength"] is None


# ===================================================================
# NoiseFilteredTrendSystem
# ===================================================================

class TestNoiseFilteredTrendSystem:

    def setup_method(self):
        self.sys = NoiseFilteredTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_clean_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_clean_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_filtered_out_when_choppy(self):
        # Last close is above its MA, but ER ~ 0 so the filter blocks it.
        closes = np.where(np.arange(100) % 2 == 0, 100.0, 102.0)
        data = _make_data(closes)
        assert closes[-1] > np.mean(closes[-20:])
        assert self.sys.efficiency_ratio(closes) < self.sys.er_threshold
        assert self.sys.signal(data) == 0
        assert self.sys.risk_filter(data) is False

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert abs(ind["efficiency_ratio"] - 1.0) < 1e-9

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["efficiency_ratio"] is None


# ===================================================================
# PriceVolumeDistributionSystem
# ===================================================================

def _cluster_closes(last):
    closes = 100.0 + (np.arange(100) % 2).astype(float)  # 100/101 cluster
    closes[-1] = last
    return closes


class TestPriceVolumeDistributionSystem:

    def setup_method(self):
        self.sys = PriceVolumeDistributionSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_above_value_area(self):
        assert self.sys.signal(_make_data(_cluster_closes(120.0))) == 1

    def test_short_below_value_area(self):
        assert self.sys.signal(_make_data(_cluster_closes(80.0))) == -1

    def test_flat_inside_value_area(self):
        assert self.sys.signal(_make_data(_cluster_closes(100.5))) == 0

    def test_flat_on_constant_price(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(_make_data(_cluster_closes(100.5)))
        assert ind["value_area_low"] < ind["value_area_high"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["value_area_low"] is None


# ===================================================================
# Cross-cutting: position sizing and risk filter
# ===================================================================

class TestPhase6CrossCutting:

    ALL = [
        AdaptiveTrendSystem(),
        AdaptiveMomentumSystem(),
        AdaptiveChannelSystem(),
        DualTimeframeTrendSystem(),
        TripleScreenSystem(),
        MultiResolutionMomentumSystem(),
        VolatilityWeightedTrendSystem(),
        NoiseFilteredTrendSystem(),
        PriceVolumeDistributionSystem(),
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
