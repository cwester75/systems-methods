"""
Test that every trading system conforms to the standardized TradingSystem
interface:

    signal(data)            → int in {-1, 0, 1}
    position_sizing(data, risk) → float >= 0
    risk_filter(data)       → bool
    indicators(data)        → dict
"""

import numpy as np
import pytest

from kaufman_systems.base import TradingSystem

from kaufman_systems.trend.er_trend_system import ERTrendSystem
from kaufman_systems.trend.linear_regression_trend import LinearRegressionTrendSystem
from kaufman_systems.moving_average.dual_ma_system import DualMASystem
from kaufman_systems.moving_average.triple_ma_system import TripleMASystem
from kaufman_systems.moving_average.kama_system import KAMASystem
from kaufman_systems.breakout.atr_breakout_system import ATRBreakoutSystem
from kaufman_systems.breakout.donchian_breakout_system import DonchianBreakoutSystem
from kaufman_systems.momentum.dual_roc_system import DualROCSystem
from kaufman_systems.momentum.rsi_reversal_system import RSIReversalSystem
from kaufman_systems.adaptive.kaufman_adaptive_system import KaufmanAdaptiveSystem
from kaufman_systems.volatility.bollinger_breakout_system import BollingerBreakoutSystem
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


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

np.random.seed(42)
_closes = np.cumsum(np.random.randn(100)) + 100
_highs = _closes + np.abs(np.random.randn(100))
_lows = _closes - np.abs(np.random.randn(100))

DATA = {"closes": _closes, "highs": _highs, "lows": _lows}
RISK = {"equity": 100_000.0}

ALL_SYSTEMS = [
    ERTrendSystem(),
    LinearRegressionTrendSystem(),
    DualMASystem(),
    TripleMASystem(),
    KAMASystem(),
    ATRBreakoutSystem(),
    DonchianBreakoutSystem(),
    DualROCSystem(),
    RSIReversalSystem(),
    KaufmanAdaptiveSystem(),
    BollingerBreakoutSystem(),
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
]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestTradingSystemInterface:

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_is_trading_system(self, system):
        assert isinstance(system, TradingSystem)

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_signal_returns_valid_int(self, system):
        sig = system.signal(DATA)
        assert sig in (-1, 0, 1), f"signal returned {sig}"

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_position_sizing_returns_float(self, system):
        size = system.position_sizing(DATA, RISK)
        assert isinstance(size, (int, float))
        assert size >= 0

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_risk_filter_returns_bool(self, system):
        result = system.risk_filter(DATA)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_indicators_returns_dict(self, system):
        ind = system.indicators(DATA)
        assert isinstance(ind, dict)

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_risk_per_trade_override(self, system):
        custom_risk = {"equity": 100_000.0, "risk_per_trade": 0.02}
        size_default = system.position_sizing(DATA, RISK)
        size_custom = system.position_sizing(DATA, custom_risk)
        # With doubled risk_per_trade, position should be ~2x (if ATR valid)
        if size_default > 0:
            assert abs(size_custom / size_default - 2.0) < 0.01

    @pytest.mark.parametrize("system", ALL_SYSTEMS, ids=lambda s: type(s).__name__)
    def test_insufficient_data_returns_safe_defaults(self, system):
        short_data = {
            "closes": np.array([100.0, 101.0]),
            "highs": np.array([101.0, 102.0]),
            "lows": np.array([99.0, 100.0]),
        }
        assert system.signal(short_data) == 0
        assert system.position_sizing(short_data, RISK) == 0
        assert system.risk_filter(short_data) is False
