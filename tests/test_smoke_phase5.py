"""
Smoke tests for the Phase 5 behavioral and cycle systems.

Each system gets its own class with crafted price data that triggers
LONG, SHORT and FLAT, plus indicator-key and insufficient-data checks.
A cross-cutting class verifies ATR-based position sizing behaviour.
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.behavioral.cot_positioning_system import COTPositioningSystem
from kaufman_systems.behavioral.news_event_system import NewsEventSystem
from kaufman_systems.behavioral.sentiment_contrarian_system import SentimentContrarianSystem
from kaufman_systems.behavioral.opinion_indicator_system import OpinionIndicatorSystem
from kaufman_systems.behavioral.fibonacci_projection_system import FibonacciProjectionSystem
from kaufman_systems.behavioral.elliott_wave_filter import ElliottWaveFilterSystem
from kaufman_systems.cycle.gann_time_cycle_system import GannTimeCycleSystem
from kaufman_systems.cycle.price_time_cycle_system import PriceTimeCycleSystem
from kaufman_systems.cycle.market_sentiment_index_system import MarketSentimentIndexSystem
from kaufman_systems.cycle.crowd_behavior_system import CrowdBehaviorSystem


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


def _zigzag(slope, n=100, amp=5, period=10):
    """Triangle wave (period bars) riding a linear trend."""
    half = period // 2
    tri = np.array([min(i % period, period - i % period) for i in range(n)]) * (amp / half)
    return 100.0 + slope * np.arange(n) + tri


# ===================================================================
# 51. COTPositioningSystem
# ===================================================================

class TestCOTPositioningSystem:

    def setup_method(self):
        self.sys = COTPositioningSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_short_on_crowded_long(self):
        # Flat, then an accelerating rally: cumulative momentum at its extreme
        closes = np.full(100, 100.0)
        closes[80:] = np.linspace(100, 130, 20)
        assert self.sys.signal(_make_data(closes)) == -1

    def test_long_on_crowded_short(self):
        closes = np.full(100, 100.0)
        closes[80:] = np.linspace(100, 70, 20)
        assert self.sys.signal(_make_data(closes)) == 1

    def test_flat_on_no_momentum(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert set(ind) == {"cumulative_momentum", "upper_threshold", "lower_threshold"}

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["cumulative_momentum"] is None


# ===================================================================
# 52. NewsEventSystem
# ===================================================================

class TestNewsEventSystem:

    def setup_method(self):
        self.sys = NewsEventSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_up_event(self):
        closes = np.full(100, 100.0)
        closes[-1] = 110.0
        assert self.sys.signal(_make_data(closes)) == 1

    def test_short_on_down_event(self):
        closes = np.full(100, 100.0)
        closes[-1] = 90.0
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_on_small_move(self):
        closes = np.full(100, 100.0)
        closes[-1] = 100.5
        assert self.sys.signal(_make_data(closes)) == 0

    def test_flat_no_event(self):
        assert self.sys.signal(DATA_UP) == 0

    def test_indicators_keys(self):
        closes = np.full(100, 100.0)
        closes[-1] = 110.0
        ind = self.sys.indicators(_make_data(closes))
        assert set(ind) == {"move", "atr", "is_event"}
        assert ind["is_event"] is True or ind["is_event"] == True  # noqa: E712

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 53. SentimentContrarianSystem
# ===================================================================

class TestSentimentContrarianSystem:

    def setup_method(self):
        self.sys = SentimentContrarianSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_short_when_overbought(self):
        # Relentless rally -> RSI 100 -> contrarian SHORT
        assert self.sys.signal(DATA_UP) == -1

    def test_long_when_oversold(self):
        assert self.sys.signal(DATA_DOWN) == 1

    def test_flat_when_neutral(self):
        closes = np.array([100.0, 101.0] * 50)
        assert self.sys.signal(_make_data(closes)) == 0

    def test_flat_market_is_not_overbought(self):
        # No gains and no losses is neutral, not RSI 100
        assert self.sys.signal(DATA_FLAT) == 0
        assert self.sys.indicators(DATA_FLAT)["rsi"] == 50.0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert set(ind) == {"rsi", "is_overbought", "is_oversold"}
        assert ind["is_overbought"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["rsi"] is None


# ===================================================================
# 54. OpinionIndicatorSystem
# ===================================================================

class TestOpinionIndicatorSystem:

    def setup_method(self):
        self.sys = OpinionIndicatorSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_short_after_up_streak(self):
        assert self.sys.signal(DATA_UP) == -1

    def test_long_after_down_streak(self):
        assert self.sys.signal(DATA_DOWN) == 1

    def test_streak_just_below_threshold_is_flat(self):
        closes = np.array([100.0, 99.0] * 30 + [100.0, 101.0, 102.0, 103.0])  # 4 up days
        assert self.sys.signal(_make_data(closes)) == 0

    def test_streak_at_threshold_fires(self):
        closes = np.array([100.0, 99.0] * 30 + [100.0, 101.0, 102.0, 103.0, 104.0])  # 5 up
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_no_streak(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert set(ind) == {"streak", "is_extreme_bullish", "is_extreme_bearish"}
        assert ind["streak"] > 0
        assert self.sys.indicators(DATA_DOWN)["streak"] < 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 55. FibonacciProjectionSystem
# ===================================================================

class TestFibonacciProjectionSystem:

    def setup_method(self):
        self.sys = FibonacciProjectionSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_above_618(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_below_382(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_between_levels(self):
        closes = np.full(100, 100.0)
        closes[60] = 200.0
        closes[-1] = 150.0  # between 138.2 and 161.8
        assert self.sys.signal(_make_data(closes)) == 0

    def test_flat_zero_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_values(self):
        closes = np.full(100, 100.0)
        closes[60] = 200.0
        ind = self.sys.indicators(_make_data(closes))
        assert set(ind) == {"swing_high", "swing_low", "fib_382", "fib_618"}
        assert ind["swing_high"] == 200.0
        assert ind["swing_low"] == 100.0
        assert ind["fib_382"] == pytest.approx(138.2)
        assert ind["fib_618"] == pytest.approx(161.8)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 56. ElliottWaveFilterSystem
# ===================================================================

class TestElliottWaveFilterSystem:

    def setup_method(self):
        self.sys = ElliottWaveFilterSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_ascending_swing_lows(self):
        assert self.sys.signal(_make_data(_zigzag(+0.5))) == 1

    def test_short_on_descending_swing_highs(self):
        assert self.sys.signal(_make_data(_zigzag(-0.5))) == -1

    def test_flat_on_no_structure(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(_make_data(_zigzag(+0.5)))
        assert set(ind) == {"swing_lows", "swing_highs", "num_swing_lows", "num_swing_highs"}
        assert ind["num_swing_lows"] >= 3
        lows = ind["swing_lows"]
        assert all(lows[i] < lows[i + 1] for i in range(len(lows) - 1))

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 57. GannTimeCycleSystem
# ===================================================================

class TestGannTimeCycleSystem:

    def setup_method(self):
        self.sys = GannTimeCycleSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_at_turn_above_ma(self):
        # 91 bars -> bar_index 90, a multiple of the 30-bar cycle
        assert self.sys.signal(_make_data(np.linspace(100, 200, 91))) == 1

    def test_short_at_turn_below_ma(self):
        assert self.sys.signal(_make_data(np.linspace(200, 100, 91))) == -1

    def test_flat_away_from_turn(self):
        # 96 bars -> bar_index 95, 5 bars from either turn
        assert self.sys.signal(_make_data(np.linspace(100, 200, 96))) == 0

    def test_flat_at_turn_price_equals_ma(self):
        assert self.sys.signal(_make_data(np.full(91, 150.0))) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(_make_data(np.linspace(100, 200, 91)))
        assert set(ind) == {"bar_index", "distance_to_cycle", "near_turn", "ma"}
        assert ind["bar_index"] == 90
        assert ind["near_turn"] is True
        assert self.sys.indicators(_make_data(np.linspace(100, 200, 96)))["near_turn"] is False

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 58. PriceTimeCycleSystem
# ===================================================================

class TestPriceTimeCyclesystem:

    def setup_method(self):
        self.sys = PriceTimeCycleSystem()

    @staticmethod
    def _cyclic(slope):
        i = np.arange(150)
        return _make_data(100.0 + slope * i + 3.0 * np.sin(2 * np.pi * i / 20))

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_detects_dominant_cycle(self):
        assert self.sys.indicators(self._cyclic(0.3))["dominant_cycle"] == 20

    def test_long_when_cycle_return_positive(self):
        assert self.sys.signal(self._cyclic(+0.3)) == 1

    def test_short_when_cycle_return_negative(self):
        assert self.sys.signal(self._cyclic(-0.3)) == -1

    def test_flat_on_constant_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0
        assert self.sys.indicators(DATA_FLAT)["dominant_cycle"] is None

    def test_indicators_keys(self):
        assert set(self.sys.indicators(DATA_UP)) == {"dominant_cycle"}

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 59. MarketSentimentIndexSystem
# ===================================================================

class TestMarketSentimentIndexSystem:

    def setup_method(self):
        self.sys = MarketSentimentIndexSystem()

    @staticmethod
    def _long_data():
        # High plateau, deep dip, small bounce: ROC > 0, z-score < -1.
        # Final five bars are wide-range so short ATR > long ATR.
        closes = np.concatenate([np.full(89, 130.0), np.full(10, 100.0), [101.0]])
        data = _make_data(closes)
        data["highs"][-5:] = closes[-5:] + 20
        data["lows"][-5:] = closes[-5:] - 20
        return data

    @staticmethod
    def _short_data():
        # Mirror image with a quiet finish so short ATR < long ATR.
        closes = np.concatenate([np.full(89, 70.0), np.full(10, 100.0), [99.0]])
        data = _make_data(closes)
        data["highs"][-5:] = closes[-5:] + 0.1
        data["lows"][-5:] = closes[-5:] - 0.1
        return data

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_bullish_composite(self):
        data = self._long_data()
        assert self.sys.indicators(data)["composite"] >= 2
        assert self.sys.signal(data) == 1

    def test_short_on_bearish_composite(self):
        data = self._short_data()
        assert self.sys.indicators(data)["composite"] <= -2
        assert self.sys.signal(data) == -1

    def test_flat_on_mixed_composite(self):
        # Steady rally: momentum +1 but z-score and volatility both -1
        assert self.sys.signal(DATA_UP) == 0

    def test_flat_on_flat_prices(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert set(ind) == {"momentum_score", "mean_reversion_score", "volatility_score", "composite"}
        assert ind["composite"] == ind["momentum_score"] + ind["mean_reversion_score"] + ind["volatility_score"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["composite"] is None


# ===================================================================
# 60. CrowdBehaviorSystem
# ===================================================================

class TestCrowdBehaviorSystem:

    def setup_method(self):
        self.sys = CrowdBehaviorSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_herding_and_above_ma(self):
        closes = np.full(100, 100.0)
        closes[-1] = 100.5
        data = _make_data(closes)
        assert self.sys.indicators(data)["is_herding"]
        assert self.sys.signal(data) == 1

    def test_short_when_herding_and_below_ma(self):
        closes = np.full(100, 100.0)
        closes[-1] = 99.5
        assert self.sys.signal(_make_data(closes)) == -1

    def test_flat_when_dispersed(self):
        # Trending prices are far from clustered
        assert self.sys.signal(DATA_UP) == 0
        assert not self.sys.indicators(DATA_UP)["is_herding"]

    def test_flat_when_price_equals_ma(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        assert set(self.sys.indicators(DATA_UP)) == {"dispersion", "ma", "is_herding"}

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.indicators(DATA_SHORT)["dispersion"] is None


# ===================================================================
# Cross-cutting
# ===================================================================

class TestPhase5PositionSizing:

    ALL = [
        COTPositioningSystem(),
        NewsEventSystem(),
        SentimentContrarianSystem(),
        OpinionIndicatorSystem(),
        FibonacciProjectionSystem(),
        ElliottWaveFilterSystem(),
        GannTimeCycleSystem(),
        PriceTimeCycleSystem(),
        MarketSentimentIndexSystem(),
        CrowdBehaviorSystem(),
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
    def test_risk_filter_true_on_normal_data(self, sys):
        assert sys.risk_filter(DATA_UP) is True

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
        assert sys.position_sizing(zero_vol, RISK) == 0
