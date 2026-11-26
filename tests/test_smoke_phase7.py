"""
Smoke tests for the 5 Phase 7 optimization / AI systems (optimization/).
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.optimization.genetic_algorithm_system import GeneticAlgorithmSystem
from kaufman_systems.optimization.neural_network_predictor import NeuralNetworkPredictor
from kaufman_systems.optimization.fuzzy_logic_system import FuzzyLogicSystem
from kaufman_systems.optimization.expert_system import ExpertSystem
from kaufman_systems.optimization.fractal_chaos_system import FractalChaosSystem


def _make_data(closes, spread=1.0):
    closes = np.asarray(closes, dtype=float)
    return {"closes": closes, "highs": closes + spread, "lows": closes - spread}


RISK = {"equity": 100_000.0}

DATA_UP = _make_data(np.linspace(100, 200, 100))
DATA_DOWN = _make_data(np.linspace(200, 100, 100))
DATA_FLAT = _make_data(np.full(100, 150.0))
DATA_SHORT = _make_data([100.0, 101.0])

# Alternating closes: strongly anti-persistent
DATA_ZIGZAG = _make_data(100.0 + 2.0 * (np.arange(100) % 2))


class TestGeneticAlgorithmSystem:

    def setup_method(self):
        self.sys = GeneticAlgorithmSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_deterministic(self):
        assert self.sys.indicators(DATA_UP) == self.sys.indicators(DATA_UP)

    def test_evolved_params_within_ranges(self):
        ind = self.sys.indicators(DATA_UP)
        assert self.sys.fast_range[0] <= ind["best_fast"] <= self.sys.fast_range[1]
        assert self.sys.slow_range[0] <= ind["best_slow"] <= self.sys.slow_range[1]

    def test_different_seed_still_valid(self):
        assert GeneticAlgorithmSystem(seed=7).signal(DATA_UP) == 1

    def test_indicators_keys_insufficient(self):
        assert self.sys.indicators(DATA_SHORT)["best_fast"] is None

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestNeuralNetworkPredictor:

    def setup_method(self):
        self.sys = NeuralNetworkPredictor()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0
        assert self.sys.indicators(DATA_FLAT)["prediction"] is None

    def test_prediction_bounded(self):
        pred = self.sys.indicators(DATA_UP)["prediction"]
        assert -1.0 <= pred <= 1.0

    def test_deterministic(self):
        assert self.sys.predict(DATA_UP["closes"]) == self.sys.predict(DATA_UP["closes"])

    def test_threshold_suppresses_signal(self):
        assert NeuralNetworkPredictor(threshold=1.0).signal(DATA_UP) == 0

    def test_non_positive_prices_are_flat(self):
        data = _make_data(np.linspace(-10, 10, 100))
        assert self.sys.signal(data) == 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestFuzzyLogicSystem:

    def setup_method(self):
        self.sys = FuzzyLogicSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_memberships_in_unit_interval(self):
        ind = self.sys.indicators(DATA_UP)
        for key in ("mu_positive", "mu_negative", "mu_high_vol"):
            assert 0.0 <= ind[key] <= 1.0

    def test_high_volatility_dampens_score(self):
        calm = _make_data(np.linspace(100, 130, 60), spread=1.0)
        highs = calm["highs"].copy()
        lows = calm["lows"].copy()
        highs[-5:] += 6.0
        lows[-5:] -= 6.0
        stormy = {"closes": calm["closes"], "highs": highs, "lows": lows}
        assert (self.sys.indicators(stormy)["score"]
                < self.sys.indicators(calm)["score"])

    def test_indicators_none_on_insufficient_data(self):
        assert self.sys.indicators(DATA_SHORT)["score"] is None

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestExpertSystem:

    def setup_method(self):
        self.sys = ExpertSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_all_rules_agree_up(self):
        assert self.sys.signal(DATA_UP) == 1
        assert self.sys.indicators(DATA_UP)["vote_total"] == 4

    def test_short_when_all_rules_agree_down(self):
        assert self.sys.signal(DATA_DOWN) == -1
        assert self.sys.indicators(DATA_DOWN)["vote_total"] == -4

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_flat_when_votes_below_minimum(self):
        assert ExpertSystem(min_votes=5).signal(DATA_UP) == 0

    def test_split_votes_are_flat(self):
        # Rally then a short pullback: slow rules bullish, momentum bearish (3 vs 1 -> net +2)
        closes = np.concatenate([np.linspace(100, 160, 90), np.linspace(160, 157, 10)])
        votes = self.sys.indicators(_make_data(closes))["votes"]
        assert len(set(votes.values())) > 1
        assert self.sys.signal(_make_data(closes)) == 0

    def test_indicators_keys(self):
        assert set(self.sys.indicators(DATA_UP)["votes"]) == {
            "trend", "momentum", "channel", "slope"}

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["votes"] is None


class TestFractalChaosSystem:

    def setup_method(self):
        self.sys = FractalChaosSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_trending_regime_follows_momentum_up(self):
        assert self.sys.indicators(DATA_UP)["regime"] == "trending"
        assert self.sys.signal(DATA_UP) == 1

    def test_trending_regime_follows_momentum_down(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_reverting_regime_detected(self):
        ind = self.sys.indicators(DATA_ZIGZAG)
        assert ind["regime"] == "reverting"
        assert ind["hurst"] < self.sys.revert_h

    def test_reverting_regime_fades_deviation(self):
        closes = 100.0 + 2.0 * (np.arange(100) % 2)
        # Make the final close sit above the SMA, so reversion says SHORT
        closes[-1] = 102.0
        assert self.sys.signal(_make_data(closes)) == -1
        closes[-1] = 100.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0
        assert self.sys.indicators(DATA_FLAT)["regime"] == "unknown"

    def test_indicators_keys(self):
        assert {"hurst", "regime"} <= set(self.sys.indicators(DATA_UP))

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestPhase7PositionSizingMath:

    ALL = [
        GeneticAlgorithmSystem(),
        NeuralNetworkPredictor(),
        FuzzyLogicSystem(),
        ExpertSystem(),
        FractalChaosSystem(),
    ]

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_scales_with_equity(self, sys):
        a = sys.position_sizing(DATA_UP, {"equity": 100_000.0})
        b = sys.position_sizing(DATA_UP, {"equity": 200_000.0})
        assert a > 0
        assert abs(b / a - 2.0) < 0.01

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_formula_is_equity_risk_over_atr(self, sys):
        atr = sys.atr(DATA_UP["highs"], DATA_UP["lows"], DATA_UP["closes"])
        expected = 100_000.0 * sys.risk_per_trade / atr
        assert sys.position_sizing(DATA_UP, RISK) == pytest.approx(expected)

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_zero_on_insufficient_data(self, sys):
        assert sys.position_sizing(DATA_SHORT, RISK) == 0

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_insufficient_data(self, sys):
        assert sys.risk_filter(DATA_SHORT) is False

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_zero_vol(self, sys):
        zero_vol = {k: np.full(100, 100.0) for k in ("closes", "highs", "lows")}
        assert sys.risk_filter(zero_vol) is False

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_true_on_normal_data(self, sys):
        assert sys.risk_filter(DATA_UP) is True
