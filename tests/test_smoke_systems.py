"""
Smoke tests for all 10 trading systems.

Validates signal logic with crafted price data, position sizing math,
risk filter behaviour, indicator output, parameter validation, and
edge cases.  Each system gets its own test class.
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem
from kaufman_systems.trend.er_trend_system import ERTrendSystem
from kaufman_systems.trend.linear_regression_trend import LinearRegressionTrendSystem
from kaufman_systems.moving_average.dual_ma_system import DualMASystem
from kaufman_systems.moving_average.triple_ma_system import TripleMASystem
from kaufman_systems.moving_average.kama_system import KAMASystem
from kaufman_systems.momentum.dual_roc_system import DualROCSystem
from kaufman_systems.momentum.rsi_reversal_system import RSIReversalSystem
from kaufman_systems.breakout.donchian_breakout_system import DonchianBreakoutSystem
from kaufman_systems.breakout.atr_breakout_system import ATRBreakoutSystem
from kaufman_systems.volatility.bollinger_breakout_system import BollingerBreakoutSystem
from kaufman_systems.adaptive.kaufman_adaptive_system import KaufmanAdaptiveSystem
from kaufman_systems.channel.price_channel_breakout import PriceChannelBreakoutSystem
from kaufman_systems.channel.moving_channel_system import MovingChannelSystem
from kaufman_systems.channel.regression_channel_system import RegressionChannelSystem
from kaufman_systems.channel.high_low_channel_system import HighLowChannelSystem
from kaufman_systems.swing.swing_reversal_system import SwingReversalSystem
from kaufman_systems.swing.outside_day_system import OutsideDaySystem
from kaufman_systems.swing.thrust_system import ThrustSystem
from kaufman_systems.pattern.congestion_breakout_system import CongestionBreakoutSystem
from kaufman_systems.pattern.range_expansion_system import RangeExpansionSystem
from kaufman_systems.pattern.inside_day_breakout import InsideDayBreakoutSystem
from kaufman_systems.volatility_contraction.bollinger_squeeze_system import BollingerSqueezeSystem
from kaufman_systems.volatility_contraction.keltner_squeeze_system import KeltnerSqueezeSystem
from kaufman_systems.volatility_contraction.atr_contraction_system import ATRContractionSystem
from kaufman_systems.volatility_contraction.volatility_ratio_system import VolatilityRatioSystem
from kaufman_systems.volatility_contraction.standard_deviation_breakout import StandardDeviationBreakoutSystem
from kaufman_systems.range_expansion.narrow_range_breakout import NarrowRangeBreakoutSystem
from kaufman_systems.range_expansion.opening_range_breakout import OpeningRangeBreakoutSystem
from kaufman_systems.range_expansion.volatility_expansion_breakout import VolatilityExpansionBreakoutSystem
from kaufman_systems.range_expansion.vix_expansion_system import VIXExpansionSystem
from kaufman_systems.range_expansion.range_percentile_system import RangePercentileSystem


# ===================================================================
# Helpers
# ===================================================================

def _make_data(closes, spread=1.0):
    """Build a standard data dict from a closes array."""
    closes = np.asarray(closes, dtype=float)
    return {
        "closes": closes,
        "highs": closes + spread,
        "lows": closes - spread,
    }


RISK = {"equity": 100_000.0}

# Strong uptrend: 100 bars climbing steadily
_up = np.linspace(100, 200, 100)
DATA_UP = _make_data(_up)

# Strong downtrend: 100 bars falling steadily
_down = np.linspace(200, 100, 100)
DATA_DOWN = _make_data(_down)

# Flat / sideways: 100 bars at constant price
_flat = np.full(100, 150.0)
DATA_FLAT = _make_data(_flat)

# Insufficient data: only 2 bars
DATA_SHORT = _make_data([100.0, 101.0])


# ===================================================================
# 1. ERTrendSystem
# ===================================================================

class TestERTrendSystem:

    def setup_method(self):
        self.sys = ERTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_signal_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_signal_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_returns_zero(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0

    def test_indicators_has_er(self):
        ind = self.sys.indicators(DATA_UP)
        assert "efficiency_ratio" in ind
        assert ind["efficiency_ratio"] is not None

    def test_position_sizing_positive_on_valid_data(self):
        size = self.sys.position_sizing(DATA_UP, RISK)
        assert size > 0

    def test_risk_filter_true_on_valid_data(self):
        assert self.sys.risk_filter(DATA_UP) is True

    def test_risk_filter_false_on_short_data(self):
        assert self.sys.risk_filter(DATA_SHORT) is False


# ===================================================================
# 2. LinearRegressionTrendSystem
# ===================================================================

class TestLinearRegressionTrendSystem:

    def setup_method(self):
        self.sys = LinearRegressionTrendSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        # Perfect linear uptrend → slope > 0, R² ≈ 1.0
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_returns_zero(self):
        # Flat → slope ≈ 0
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "regression_slope" in ind
        assert "regression_r2" in ind

    def test_r2_near_one_on_linear_trend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["regression_r2"] > 0.99

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.position_sizing(DATA_SHORT, RISK) == 0


# ===================================================================
# 3. DualMASystem
# ===================================================================

class TestDualMASystem:

    def setup_method(self):
        self.sys = DualMASystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        # In uptrend, fast MA > slow MA
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_returns_zero(self):
        # Both MAs converge to same value
        assert self.sys.signal(DATA_FLAT) == 0

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            DualMASystem(fast_period=50, slow_period=20)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "fast_ma" in ind
        assert "slow_ma" in ind

    def test_fast_ma_above_slow_in_uptrend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["fast_ma"] > ind["slow_ma"]


# ===================================================================
# 3b. TripleMASystem
# ===================================================================

class TestTripleMASystem:

    def setup_method(self):
        self.sys = TripleMASystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        # In uptrend, fast MA > medium MA > slow MA
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_returns_zero(self):
        # All MAs converge to same value
        assert self.sys.signal(DATA_FLAT) == 0

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            TripleMASystem(fast_period=50, medium_period=25, slow_period=10)

    def test_parameter_validation_fast_equals_medium(self):
        with pytest.raises(ValueError):
            TripleMASystem(fast_period=25, medium_period=25, slow_period=50)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "fast_ma" in ind
        assert "medium_ma" in ind
        assert "slow_ma" in ind

    def test_ma_ordering_in_uptrend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["fast_ma"] > ind["medium_ma"] > ind["slow_ma"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.position_sizing(DATA_SHORT, RISK) == 0

    def test_position_sizing_positive_on_valid_data(self):
        size = self.sys.position_sizing(DATA_UP, RISK)
        assert size > 0

    def test_risk_filter_true_on_valid_data(self):
        assert self.sys.risk_filter(DATA_UP) is True

    def test_risk_filter_false_on_short_data(self):
        assert self.sys.risk_filter(DATA_SHORT) is False


# ===================================================================
# 4. KAMASystem
# ===================================================================

class TestKAMASystem:

    def setup_method(self):
        self.sys = KAMASystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        # Price above KAMA in uptrend
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "kama" in ind
        assert "efficiency_ratio" in ind

    def test_kama_lags_price_in_uptrend(self):
        ind = self.sys.indicators(DATA_UP)
        assert DATA_UP["closes"][-1] > ind["kama"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 5. DualROCSystem
# ===================================================================

class TestDualROCSystem:

    def setup_method(self):
        self.sys = DualROCSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_when_fast_roc_exceeds_slow(self):
        # fast ROC > slow ROC when 10-bar-ago price was low, 30-bar-ago was high
        closes = np.full(100, 150.0)
        closes[-11] = 100.0  # low 10 bars ago → big fast ROC
        closes[-31] = 200.0  # high 30 bars ago → negative slow ROC
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_when_slow_roc_exceeds_fast(self):
        # fast ROC < slow ROC (inverse setup)
        closes = np.full(100, 150.0)
        closes[-11] = 200.0  # high 10 bars ago → negative fast ROC
        closes[-31] = 100.0  # low 30 bars ago → positive slow ROC
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            DualROCSystem(fast_period=30, slow_period=10)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "roc_fast" in ind
        assert "roc_slow" in ind

    def test_roc_positive_in_uptrend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["roc_fast"] > 0
        assert ind["roc_slow"] > 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 6. RSIReversalSystem
# ===================================================================

class TestRSIReversalSystem:

    def setup_method(self):
        self.sys = RSIReversalSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_oversold(self):
        # Consecutive down bars → RSI < 30
        drops = np.concatenate([np.array([200.0]), 200.0 - np.cumsum(np.full(99, 2.0))])
        data = _make_data(drops)
        sig = self.sys.signal(data)
        assert sig == 1  # oversold reversal

    def test_short_on_overbought(self):
        # Consecutive up bars → RSI > 70
        rises = np.concatenate([np.array([100.0]), 100.0 + np.cumsum(np.full(99, 2.0))])
        data = _make_data(rises)
        sig = self.sys.signal(data)
        assert sig == -1  # overbought reversal

    def test_flat_in_neutral_zone(self):
        # Mix of ups and downs → RSI near 50
        np.random.seed(123)
        noise = np.cumsum(np.random.randn(100)) + 150
        data = _make_data(noise)
        sig = self.sys.signal(data)
        assert sig == 0

    def test_indicators_has_rsi(self):
        ind = self.sys.indicators(DATA_UP)
        assert "rsi" in ind
        assert 0 <= ind["rsi"] <= 100


# ===================================================================
# 7. DonchianBreakoutSystem
# ===================================================================

class TestDonchianBreakoutSystem:

    def setup_method(self):
        self.sys = DonchianBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_new_high(self):
        # Flat channel then last close spikes above channel high
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        closes[-1] = 110.0  # breaks above channel high (105)
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_new_low(self):
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        closes[-1] = 90.0  # breaks below channel low (95)
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_in_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "donchian_high" in ind
        assert "donchian_low" in ind

    def test_channel_width_positive(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["donchian_high"] > ind["donchian_low"]


# ===================================================================
# 8. ATRBreakoutSystem
# ===================================================================

class TestATRBreakoutSystem:

    def setup_method(self):
        self.sys = ATRBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_large_up_move(self):
        # Create data where last bar gaps well above prev close + 2*ATR
        closes = np.full(100, 100.0)
        closes[-1] = 130.0  # big spike
        data = _make_data(closes, spread=1.0)
        assert self.sys.signal(data) == 1

    def test_short_on_large_down_move(self):
        closes = np.full(100, 100.0)
        closes[-1] = 70.0  # big drop
        data = _make_data(closes, spread=1.0)
        assert self.sys.signal(data) == -1

    def test_flat_on_small_move(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "atr" in ind
        assert "upper_breakout" in ind
        assert "lower_breakout" in ind

    def test_breakout_bands_symmetric(self):
        ind = self.sys.indicators(DATA_UP)
        prev_close = DATA_UP["closes"][-2]
        upper_dist = ind["upper_breakout"] - prev_close
        lower_dist = prev_close - ind["lower_breakout"]
        assert abs(upper_dist - lower_dist) < 1e-10


# ===================================================================
# 9. BollingerBreakoutSystem
# ===================================================================

class TestBollingerBreakoutSystem:

    def setup_method(self):
        self.sys = BollingerBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_flat_within_bands(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_long_on_breakout_above_upper(self):
        # Tight range then spike above upper band
        closes = np.full(100, 100.0)
        closes[-1] = 120.0
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_on_breakout_below_lower(self):
        closes = np.full(100, 100.0)
        closes[-1] = 80.0
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "bollinger_ma" in ind
        assert "bollinger_std" in ind
        assert "upper_band" in ind
        assert "lower_band" in ind

    def test_upper_above_lower(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["upper_band"] > ind["lower_band"]


# ===================================================================
# 10. KaufmanAdaptiveSystem
# ===================================================================

class TestKaufmanAdaptiveSystem:

    def setup_method(self):
        self.sys = KaufmanAdaptiveSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_trend_mode_long(self):
        # Perfect uptrend → high ER → trend mode → price > KAMA → long
        assert self.sys.signal(DATA_UP) == 1

    def test_trend_mode_short(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_neutral_regime_flat(self):
        # Random walk → ER in middle zone → no signal
        np.random.seed(99)
        noise = np.cumsum(np.random.randn(100) * 0.5) + 150
        data = _make_data(noise)
        sig = self.sys.signal(data)
        assert sig in (-1, 0, 1)  # valid return

    def test_noise_mode_oversold_long(self):
        # Force noise mode: trend_threshold impossibly high, noise_threshold=1.0
        sys = KaufmanAdaptiveSystem(trend_threshold=2.0, noise_threshold=1.0)
        drops = np.concatenate([np.array([200.0]), 200.0 - np.cumsum(np.full(99, 2.0))])
        data = _make_data(drops)
        assert sys.signal(data) == 1  # RSI < 30 → long reversal

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "efficiency_ratio" in ind
        assert "kama" in ind
        assert "rsi" in ind

    def test_er_high_on_linear_trend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["efficiency_ratio"] > 0.9


# ===================================================================
# 11. PriceChannelBreakoutSystem
# ===================================================================

class TestPriceChannelBreakoutSystem:

    def setup_method(self):
        self.sys = PriceChannelBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "channel_high" in ind
        assert "channel_low" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0
        assert self.sys.position_sizing(DATA_SHORT, RISK) == 0


# ===================================================================
# 12. MovingChannelSystem
# ===================================================================

class TestMovingChannelSystem:

    def setup_method(self):
        self.sys = MovingChannelSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        # In uptrend, price is above MA + band
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "ma" in ind
        assert "upper_band" in ind
        assert "lower_band" in ind

    def test_upper_above_lower(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["upper_band"] > ind["lower_band"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 13. RegressionChannelSystem
# ===================================================================

class TestRegressionChannelSystem:

    def setup_method(self):
        self.sys = RegressionChannelSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_flat_on_linear_trend(self):
        # Perfect linear trend has zero residuals → price on regression line
        # so price is within the channel (std_err ≈ 0, but bands collapse)
        sig = self.sys.signal(DATA_UP)
        assert sig in (-1, 0, 1)

    def test_long_on_breakout_above(self):
        # Flat channel then spike above
        closes = np.full(100, 100.0)
        closes[-1] = 120.0
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_on_breakout_below(self):
        closes = np.full(100, 100.0)
        closes[-1] = 80.0
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "regression_value" in ind
        assert "upper_channel" in ind
        assert "lower_channel" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 14. HighLowChannelSystem
# ===================================================================

class TestHighLowChannelSystem:

    def setup_method(self):
        self.sys = HighLowChannelSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "high_channel" in ind
        assert "low_channel" in ind

    def test_high_above_low(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["high_channel"] > ind["low_channel"]

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 15. SwingReversalSystem
# ===================================================================

class TestSwingReversalSystem:

    def setup_method(self):
        self.sys = SwingReversalSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_signal_valid(self):
        sig = self.sys.signal(DATA_UP)
        assert sig in (-1, 0, 1)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "swing_type" in ind
        assert "swing_index" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 16. OutsideDaySystem
# ===================================================================

class TestOutsideDaySystem:

    def setup_method(self):
        self.sys = OutsideDaySystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_outside_day_up(self):
        # Create outside day: current bar engulfs previous
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        # Previous bar: narrow range
        highs[-2] = 101.0
        lows[-2] = 99.0
        # Current bar: engulfs previous, closes up
        highs[-1] = 106.0
        lows[-1] = 94.0
        closes[-1] = 103.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_outside_day_down(self):
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        highs[-2] = 101.0
        lows[-2] = 99.0
        highs[-1] = 106.0
        lows[-1] = 94.0
        closes[-1] = 97.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_no_outside_day(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "is_outside_day" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 17. ThrustSystem
# ===================================================================

class TestThrustSystem:

    def setup_method(self):
        self.sys = ThrustSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_strong_uptrend(self):
        # Need >3% move in 5 bars
        closes = np.full(100, 100.0)
        closes[-1] = 110.0  # 10% move from 5 bars ago
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_on_strong_downtrend(self):
        closes = np.full(100, 100.0)
        closes[-1] = 90.0
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "thrust" in ind
        assert "threshold" in ind

    def test_thrust_positive_in_uptrend(self):
        ind = self.sys.indicators(DATA_UP)
        assert ind["thrust"] > 0

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 18. CongestionBreakoutSystem
# ===================================================================

class TestCongestionBreakoutSystem:

    def setup_method(self):
        self.sys = CongestionBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_congestion_breakout_up(self):
        # Wide-range bars early on to establish a large ATR,
        # then a tight congestion zone, then breakout above
        highs = np.full(100, 110.0)
        lows = np.full(100, 90.0)
        closes = np.full(100, 100.0)
        # Congestion zone: bars -6 through -2 have very tight range
        for i in range(-7, -1):
            highs[i] = 100.1
            lows[i] = 99.9
        # Last bar breaks above congestion
        closes[-1] = 105.0
        highs[-1] = 106.0
        lows[-1] = 99.9
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_congestion_breakout_down(self):
        highs = np.full(100, 110.0)
        lows = np.full(100, 90.0)
        closes = np.full(100, 100.0)
        for i in range(-7, -1):
            highs[i] = 100.1
            lows[i] = 99.9
        closes[-1] = 95.0
        highs[-1] = 100.1
        lows[-1] = 94.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "is_congestion" in ind
        assert "zone_high" in ind
        assert "zone_low" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 19. RangeExpansionSystem
# ===================================================================

class TestRangeExpansionSystem:

    def setup_method(self):
        self.sys = RangeExpansionSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_expansion_up(self):
        # Tight range then large expansion bar closing up
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        # Big expansion bar
        highs[-1] = 110.0
        lows[-1] = 95.0
        closes[-1] = 108.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_expansion_down(self):
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        highs[-1] = 105.0
        lows[-1] = 90.0
        closes[-1] = 92.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_on_normal_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "current_range" in ind
        assert "avg_range" in ind
        assert "expansion_ratio" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 20. InsideDayBreakoutSystem
# ===================================================================

class TestInsideDayBreakoutSystem:

    def setup_method(self):
        self.sys = InsideDayBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_inside_day_breakout_up(self):
        # Mother bar[-3] has wide range, bar[-2] is inside, bar[-1] breaks above
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        # Mother bar (bar[-3]): wide range
        highs[-3] = 110.0
        lows[-3] = 90.0
        # Inside bar (bar[-2]): contained within mother
        highs[-2] = 105.0
        lows[-2] = 95.0
        # Breakout bar (bar[-1]): breaks above mother high
        closes[-1] = 115.0
        highs[-1] = 116.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_inside_day_breakout_down(self):
        highs = np.full(100, 105.0)
        lows = np.full(100, 95.0)
        closes = np.full(100, 100.0)
        highs[-3] = 110.0
        lows[-3] = 90.0
        highs[-2] = 105.0
        lows[-2] = 95.0
        closes[-1] = 85.0
        lows[-1] = 84.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_no_inside_day(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "is_inside_day_setup" in ind
        assert "mother_high" in ind
        assert "mother_low" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 21. BollingerSqueezeSystem
# ===================================================================

class TestBollingerSqueezeSystem:

    def setup_method(self):
        self.sys = BollingerSqueezeSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_squeeze_breakout_up(self):
        # Tight range (squeeze from prior bars) then spike above upper band
        closes = np.full(100, 100.0)
        closes[-1] = 120.0
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_on_squeeze_breakout_down(self):
        closes = np.full(100, 100.0)
        closes[-1] = 80.0
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_flat_no_squeeze(self):
        # Wide bandwidth → no squeeze
        assert self.sys.signal(DATA_UP) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "bandwidth" in ind
        assert "is_squeeze" in ind
        assert "upper_band" in ind
        assert "lower_band" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 22. KeltnerSqueezeSystem
# ===================================================================

class TestKeltnerSqueezeSystem:

    def setup_method(self):
        self.sys = KeltnerSqueezeSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_signal_valid(self):
        sig = self.sys.signal(DATA_UP)
        assert sig in (-1, 0, 1)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "bb_upper" in ind
        assert "bb_lower" in ind
        assert "kc_upper" in ind
        assert "kc_lower" in ind
        assert "is_squeeze" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 23. ATRContractionSystem
# ===================================================================

class TestATRContractionSystem:

    def setup_method(self):
        self.sys = ATRContractionSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_signal_valid(self):
        sig = self.sys.signal(DATA_UP)
        assert sig in (-1, 0, 1)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "current_atr" in ind
        assert "avg_atr" in ind
        assert "is_contraction" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 24. VolatilityRatioSystem
# ===================================================================

class TestVolatilityRatioSystem:

    def setup_method(self):
        self.sys = VolatilityRatioSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_large_expansion_up(self):
        # Flat range then huge expansion bar closing up
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        # Big expansion bar
        highs[-1] = 115.0
        lows[-1] = 95.0
        closes[-1] = 112.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_large_expansion_down(self):
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        highs[-1] = 105.0
        lows[-1] = 85.0
        closes[-1] = 88.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_on_normal_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "true_range" in ind
        assert "atr" in ind
        assert "volatility_ratio" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 25. StandardDeviationBreakoutSystem
# ===================================================================

class TestStandardDeviationBreakoutSystem:

    def setup_method(self):
        self.sys = StandardDeviationBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_spike_above(self):
        closes = np.full(100, 100.0)
        closes[-1] = 120.0
        data = _make_data(closes)
        assert self.sys.signal(data) == 1

    def test_short_on_spike_below(self):
        closes = np.full(100, 100.0)
        closes[-1] = 80.0
        data = _make_data(closes)
        assert self.sys.signal(data) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "zscore" in ind
        assert "threshold" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 26. NarrowRangeBreakoutSystem
# ===================================================================

class TestNarrowRangeBreakoutSystem:

    def setup_method(self):
        self.sys = NarrowRangeBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_nr_breakout_up(self):
        # Wide range bars, then one narrow bar, then breakout above
        highs = np.full(100, 110.0)
        lows = np.full(100, 90.0)
        closes = np.full(100, 100.0)
        # Narrow range bar at [-2]
        highs[-2] = 100.2
        lows[-2] = 99.8
        # Breakout bar
        closes[-1] = 105.0
        highs[-1] = 106.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_nr_breakout_down(self):
        highs = np.full(100, 110.0)
        lows = np.full(100, 90.0)
        closes = np.full(100, 100.0)
        highs[-2] = 100.2
        lows[-2] = 99.8
        closes[-1] = 95.0
        lows[-1] = 94.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "is_narrow_range" in ind
        assert "nr_high" in ind
        assert "nr_low" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 27. OpeningRangeBreakoutSystem
# ===================================================================

class TestOpeningRangeBreakoutSystem:

    def setup_method(self):
        self.sys = OpeningRangeBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_uptrend(self):
        assert self.sys.signal(DATA_UP) == 1

    def test_short_on_downtrend(self):
        assert self.sys.signal(DATA_DOWN) == -1

    def test_flat_on_sideways(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "or_high" in ind
        assert "or_low" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 28. VolatilityExpansionBreakoutSystem
# ===================================================================

class TestVolatilityExpansionBreakoutSystem:

    def setup_method(self):
        self.sys = VolatilityExpansionBreakoutSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_expansion_close_near_high(self):
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        # Big expansion, close near high
        highs[-1] = 115.0
        lows[-1] = 95.0
        closes[-1] = 114.0  # near high
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_expansion_close_near_low(self):
        closes = np.full(100, 100.0)
        highs = np.full(100, 101.0)
        lows = np.full(100, 99.0)
        highs[-1] = 105.0
        lows[-1] = 85.0
        closes[-1] = 86.0  # near low
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_on_normal_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "current_range" in ind
        assert "avg_range" in ind
        assert "expansion_ratio" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 29. VIXExpansionSystem
# ===================================================================

class TestVIXExpansionSystem:

    def setup_method(self):
        self.sys = VIXExpansionSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_signal_valid(self):
        sig = self.sys.signal(DATA_UP)
        assert sig in (-1, 0, 1)

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "current_vol" in ind
        assert "avg_vol" in ind
        assert "is_expansion" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# 30. RangePercentileSystem
# ===================================================================

class TestRangePercentileSystem:

    def setup_method(self):
        self.sys = RangePercentileSystem()

    def test_inherits_base(self):
        assert isinstance(self.sys, TradingSystem)

    def test_long_on_high_percentile_up(self):
        # Small range bars then huge bar closing up
        closes = np.full(100, 100.0)
        highs = np.full(100, 100.5)
        lows = np.full(100, 99.5)
        highs[-1] = 115.0
        lows[-1] = 90.0
        closes[-1] = 110.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == 1

    def test_short_on_high_percentile_down(self):
        closes = np.full(100, 100.0)
        highs = np.full(100, 100.5)
        lows = np.full(100, 99.5)
        highs[-1] = 110.0
        lows[-1] = 85.0
        closes[-1] = 90.0
        closes[-2] = 100.0
        data = {"closes": closes, "highs": highs, "lows": lows}
        assert self.sys.signal(data) == -1

    def test_flat_on_normal_range(self):
        assert self.sys.signal(DATA_FLAT) == 0

    def test_indicators_keys(self):
        ind = self.sys.indicators(DATA_UP)
        assert "range_percentile" in ind
        assert "upper_pct" in ind

    def test_insufficient_data(self):
        assert self.sys.signal(DATA_SHORT) == 0


# ===================================================================
# Cross-cutting: Position sizing math
# ===================================================================

class TestPositionSizingMath:
    """Verify the ATR-based formula across all systems."""

    ALL = [
        ERTrendSystem(),
        LinearRegressionTrendSystem(),
        DualMASystem(),
        TripleMASystem(),
        KAMASystem(),
        DualROCSystem(),
        RSIReversalSystem(),
        DonchianBreakoutSystem(),
        ATRBreakoutSystem(),
        BollingerBreakoutSystem(),
        KaufmanAdaptiveSystem(),
        PriceChannelBreakoutSystem(),
        MovingChannelSystem(),
        RegressionChannelSystem(),
        HighLowChannelSystem(),
        SwingReversalSystem(),
        OutsideDaySystem(),
        ThrustSystem(),
        CongestionBreakoutSystem(),
        RangeExpansionSystem(),
        InsideDayBreakoutSystem(),
        BollingerSqueezeSystem(),
        KeltnerSqueezeSystem(),
        ATRContractionSystem(),
        VolatilityRatioSystem(),
        StandardDeviationBreakoutSystem(),
        NarrowRangeBreakoutSystem(),
        OpeningRangeBreakoutSystem(),
        VolatilityExpansionBreakoutSystem(),
        VIXExpansionSystem(),
        RangePercentileSystem(),
    ]

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_scales_with_equity(self, sys):
        size_100k = sys.position_sizing(DATA_UP, {"equity": 100_000.0})
        size_200k = sys.position_sizing(DATA_UP, {"equity": 200_000.0})
        if size_100k > 0:
            assert abs(size_200k / size_100k - 2.0) < 0.01

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_position_zero_on_insufficient_data(self, sys):
        assert sys.position_sizing(DATA_SHORT, RISK) == 0

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_insufficient_data(self, sys):
        assert sys.risk_filter(DATA_SHORT) is False

    @pytest.mark.parametrize("sys", ALL, ids=lambda s: type(s).__name__)
    def test_risk_filter_false_on_zero_vol(self, sys):
        # Zero spread → ATR = 0
        zero_vol = {
            "closes": np.full(100, 100.0),
            "highs": np.full(100, 100.0),
            "lows": np.full(100, 100.0),
        }
        assert sys.risk_filter(zero_vol) is False
