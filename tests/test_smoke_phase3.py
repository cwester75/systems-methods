"""
Smoke tests for the 10 Phase 3 pattern systems (patterns/).

Crafted price data triggers each system's signal condition; a cross-cutting
class checks position sizing and risk filter behaviour for all of them.
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.patterns.opening_gap_system import OpeningGapSystem
from kaufman_systems.patterns.weekend_effect_system import WeekendEffectSystem
from kaufman_systems.patterns.weekday_pattern_system import WeekdayPatternSystem
from kaufman_systems.patterns.reversal_pattern_system import ReversalPatternSystem
from kaufman_systems.patterns.time_of_day_pattern import TimeOfDayPatternSystem
from kaufman_systems.patterns.intraday_range_pattern import IntradayRangePatternSystem
from kaufman_systems.patterns.seasonal_pattern_system import SeasonalPatternSystem
from kaufman_systems.patterns.monthly_turn_system import MonthlyTurnSystem
from kaufman_systems.patterns.holiday_effect_system import HolidayEffectSystem
from kaufman_systems.patterns.earnings_drift_system import EarningsDriftSystem


def _make_data(closes, spread=1.0):
    closes = np.asarray(closes, dtype=float)
    return {"closes": closes, "highs": closes + spread, "lows": closes - spread}


RISK = {"equity": 100_000.0}

DATA_UP = _make_data(np.linspace(100, 200, 100))
DATA_DOWN = _make_data(np.linspace(200, 100, 100))
DATA_FLAT = _make_data(np.full(100, 150.0))
DATA_SHORT = _make_data([100.0, 101.0])


# -------------------------------------------------------------------
# OpeningGapSystem
# -------------------------------------------------------------------

class TestOpeningGapSystem:

    def setup_method(self):
        self.sys = OpeningGapSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_gap_up(self):
        # Flat 100 with ATR ~2, then a bar whose low sits far above prior close
        closes = np.full(30, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        closes[-1], highs[-1], lows[-1] = 120.0, 121.0, 115.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_gap_down(self):
        closes = np.full(30, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        closes[-1], highs[-1], lows[-1] = 80.0, 85.0, 79.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_no_gap(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"atr", "gap_up", "gap_down"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# WeekendEffectSystem / WeekdayPatternSystem (5-bar cycle return bias)
# -------------------------------------------------------------------

def _cycle_data(n, up_pos, drift, cycle=5):
    """Series whose return is +drift at cycle position `up_pos` of the
    returns array and 0 elsewhere (position = index within returns)."""
    rets = np.zeros(n - 1)
    rets[np.arange(n - 1) % cycle == up_pos] = drift
    closes = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    return _make_data(closes, spread=0.5)


class TestWeekendEffectSystem:

    def setup_method(self):
        self.sys = WeekendEffectSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def _current_pos(self):
        # signal() uses the last `lookback` returns; current position is
        # (lookback - 1) % cycle_length within that window
        return (self.sys.lookback - 1) % self.sys.cycle_length

    def test_long_when_current_position_is_positive(self):
        n = self.sys.lookback + 1
        data = _cycle_data(n, self._current_pos(), 0.02)
        assert self.sys.signal(data) == 1

    def test_short_when_current_position_is_negative(self):
        n = self.sys.lookback + 1
        data = _cycle_data(n, self._current_pos(), -0.02)
        assert self.sys.signal(data) == -1

    def test_flat_when_no_bias(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"cycle_position", "avg_return_at_position"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


class TestWeekdayPatternSystem:

    def setup_method(self):
        self.sys = WeekdayPatternSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def _current_pos(self):
        return (self.sys.lookback - 1) % self.sys.cycle_length

    def test_long_when_current_position_is_positive(self):
        data = _cycle_data(self.sys.lookback + 1, self._current_pos(), 0.02)
        assert self.sys.signal(data) == 1

    def test_short_when_current_position_is_negative(self):
        data = _cycle_data(self.sys.lookback + 1, self._current_pos(), -0.02)
        assert self.sys.signal(data) == -1

    def test_flat_when_no_bias(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"cycle_position", "avg_return_at_position"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# ReversalPatternSystem
# -------------------------------------------------------------------

class TestReversalPatternSystem:

    def setup_method(self):
        self.sys = ReversalPatternSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_bullish_key_reversal(self):
        closes = np.full(30, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        # New low below the prior 20-bar low, but close above prior close
        lows[-1] = 90.0
        closes[-1] = 101.0
        highs[-1] = 102.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1
        assert self.sys.indicators(data)["bullish_reversal"] is True

    def test_short_on_bearish_key_reversal(self):
        closes = np.full(30, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        highs[-1] = 110.0
        closes[-1] = 99.0
        lows[-1] = 98.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1
        assert self.sys.indicators(data)["bearish_reversal"] is True

    def test_flat_no_reversal(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# TimeOfDayPatternSystem
# -------------------------------------------------------------------

class TestTimeOfDayPatternSystem:

    def setup_method(self):
        self.sys = TimeOfDayPatternSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_close_in_upper_range(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_close_in_lower_range(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_zero_span(self):
        zero = {k: np.full(30, 100.0) for k in ("closes", "highs", "lows")}
        assert self.sys.signal(zero) == 0

    def test_flat_mid_range(self):
        closes = np.full(30, 100.0)
        highs = closes + 10.0
        lows = closes - 10.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"range_high", "range_low", "pct_position"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# IntradayRangePatternSystem
# -------------------------------------------------------------------

class TestIntradayRangePatternSystem:

    def setup_method(self):
        self.sys = IntradayRangePatternSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def _expansion_data(self, close):
        closes = np.full(30, 100.0)
        highs = closes + 1.0
        lows = closes - 1.0
        highs[-1], lows[-1], closes[-1] = 110.0, 90.0, close
        return {"closes": closes, "highs": highs, "lows": lows}

    def test_long_on_expansion_closing_high(self):
        assert self.sys.signal(self._expansion_data(108.0)) == 1

    def test_short_on_expansion_closing_low(self):
        assert self.sys.signal(self._expansion_data(92.0)) == -1

    def test_flat_without_expansion(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"avg_range", "current_range", "expansion_ratio"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# SeasonalPatternSystem
# -------------------------------------------------------------------

class TestSeasonalPatternSystem:

    def setup_method(self):
        self.sys = SeasonalPatternSystem(cycle_period=50)

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_above_cycle_ago(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_when_below_cycle_ago(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_when_unchanged(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["cycle_period"] == 50
        assert ind["seasonal_return"] > 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0

    def test_default_needs_full_cycle(self):
        # Default 252-bar cycle: 100 bars is not enough
        assert SeasonalPatternSystem().signal(DATA_UP) == 0


# -------------------------------------------------------------------
# MonthlyTurnSystem
# -------------------------------------------------------------------

class TestMonthlyTurnSystem:

    def setup_method(self):
        self.sys = MonthlyTurnSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    @pytest.mark.parametrize("n", [21 * 4 + 1, 21 * 4 + 2, 21 * 4 + 3, 21 * 4 + 20, 21 * 4 + 19])
    def test_long_at_turn_positions(self, n):
        # bar_position = (n - 1) % 21: 0,1,2 and 19,20 are turn bars
        closes = np.linspace(100, 110, n)
        assert self.sys.signal(_make_data(closes)) == 1

    def test_flat_mid_month(self):
        n = 21 * 4 + 11  # position 10
        closes = np.linspace(100, 110, n)
        assert self.sys.signal(_make_data(closes)) == 0

    def test_never_short(self):
        for n in range(20, 120):
            closes = np.linspace(110, 100, n)
            assert self.sys.signal(_make_data(closes)) >= 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"bar_position", "is_turn_of_month"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# HolidayEffectSystem
# -------------------------------------------------------------------

class TestHolidayEffectSystem:

    def setup_method(self):
        self.sys = HolidayEffectSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_volatility_contraction(self):
        # Wide bars for most of the history, then 5 narrow bars
        closes = np.full(40, 100.0)
        highs = closes + 5.0
        lows = closes - 5.0
        highs[-5:] = 100.2
        lows[-5:] = 99.8
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1
        assert self.sys.indicators(data)["is_contraction"]

    def test_flat_without_contraction(self):
        assert self.sys.signal(DATA_UP) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert {"short_atr", "long_atr", "atr_ratio", "is_contraction"} <= set(ind)

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# EarningsDriftSystem
# -------------------------------------------------------------------

class TestEarningsDriftSystem:

    def setup_method(self):
        self.sys = EarningsDriftSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def _event_data(self, jump):
        # Event on bar[-2]; bar[-1] is the continuation bar
        closes = np.full(30, 100.0)
        closes[-2:] = 100.0 + jump
        return _make_data(closes)

    def test_long_after_large_up_move(self):
        assert self.sys.signal(self._event_data(+15.0)) == 1

    def test_short_after_large_down_move(self):
        assert self.sys.signal(self._event_data(-15.0)) == -1

    def test_flat_no_event(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(self._event_data(15.0))
        assert {"atr", "event_move", "is_event"} <= set(ind)
        assert ind["is_event"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# -------------------------------------------------------------------
# Cross-cutting
# -------------------------------------------------------------------

class TestPhase3PositionSizingMath:

    ALL = [
        OpeningGapSystem(),
        WeekendEffectSystem(),
        WeekdayPatternSystem(),
        ReversalPatternSystem(),
        TimeOfDayPatternSystem(),
        IntradayRangePatternSystem(),
        SeasonalPatternSystem(),
        MonthlyTurnSystem(),
        HolidayEffectSystem(),
        EarningsDriftSystem(),
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
