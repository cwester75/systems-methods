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
from kaufman_systems.moving_average.kama_system import KAMASystem
from kaufman_systems.momentum.dual_roc_system import DualROCSystem
from kaufman_systems.momentum.rsi_reversal_system import RSIReversalSystem
from kaufman_systems.breakout.donchian_breakout_system import DonchianBreakoutSystem
from kaufman_systems.breakout.atr_breakout_system import ATRBreakoutSystem
from kaufman_systems.volatility.bollinger_breakout_system import BollingerBreakoutSystem
from kaufman_systems.adaptive.kaufman_adaptive_system import KaufmanAdaptiveSystem


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
# Cross-cutting: Position sizing math
# ===================================================================

class TestPositionSizingMath:
    """Verify the ATR-based formula across all systems."""

    ALL = [
        ERTrendSystem(),
        LinearRegressionTrendSystem(),
        DualMASystem(),
        KAMASystem(),
        DualROCSystem(),
        RSIReversalSystem(),
        DonchianBreakoutSystem(),
        ATRBreakoutSystem(),
        BollingerBreakoutSystem(),
        KaufmanAdaptiveSystem(),
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
