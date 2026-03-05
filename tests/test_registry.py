"""Tests for the Indicator Registry."""

import numpy as np
import pytest

from kaufman_indicators.registry import INDICATORS, get


class TestRegistry:
    def test_all_entries_are_callable(self):
        for name, fn in INDICATORS.items():
            assert callable(fn), f"{name} is not callable"

    def test_expected_count(self):
        # 9 trend + 5 momentum + 5 volatility + 4 range + 5 market_quality = 28
        assert len(INDICATORS) == 28

    def test_get_returns_function(self):
        fn = get("rsi")
        assert fn is INDICATORS["rsi"]

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown indicator"):
            get("nonexistent")

    def test_get_error_lists_available(self):
        with pytest.raises(KeyError, match="rsi"):
            get("nonexistent")

    def test_call_single_array_indicators(self):
        """Smoke-test indicators that take a single price array."""
        prices = np.linspace(100, 150, 50)
        # Indicators with default period that accept (prices) or (prices, period=...)
        single_array = [
            "efficiency_ratio", "kama", "linreg", "linreg_forecast",
            "roc", "rsi", "momentum",
            "fdi", "hurst_exponent", "price_entropy",
            "price_zscore", "volume_roc", "volume_zscore",
        ]
        for name in single_array:
            fn = get(name)
            result = fn(prices)
            if isinstance(result, tuple):
                assert len(result[0]) == 50, f"{name} length mismatch"
            else:
                assert len(result) == 50, f"{name} length mismatch"

    def test_call_moving_averages(self):
        """Smoke-test moving averages that require an explicit period."""
        prices = np.linspace(100, 150, 50)
        for name in ["sma", "ema", "wma", "dema", "tema"]:
            result = get(name)(prices, period=10)
            assert len(result) == 50, f"{name} length mismatch"

    def test_call_macd(self):
        prices = np.linspace(100, 150, 50)
        result = get("macd")(prices)
        assert len(result.macd_line) == 50

    def test_call_atr(self):
        high = np.linspace(105, 155, 50)
        low = np.linspace(95, 145, 50)
        close = np.linspace(100, 150, 50)
        result = get("atr")(high, low, close)
        assert len(result) == 50

    def test_call_bollinger(self):
        prices = np.linspace(100, 150, 50)
        result = get("bollinger_bands")(prices)
        assert len(result.middle) == 50

    def test_call_donchian(self):
        high = np.linspace(105, 155, 50)
        low = np.linspace(95, 145, 50)
        result = get("donchian_channels")(high, low)
        assert len(result.upper) == 50

    def test_call_stochastic(self):
        high = np.linspace(105, 155, 50)
        low = np.linspace(95, 145, 50)
        close = np.linspace(100, 150, 50)
        result = get("stochastic")(high, low, close)
        assert len(result.k) == 50

    def test_call_williams_r(self):
        high = np.linspace(105, 155, 50)
        low = np.linspace(95, 145, 50)
        close = np.linspace(100, 150, 50)
        result = get("williams_r")(high, low, close)
        assert len(result) == 50

    def test_import_from_package(self):
        from kaufman_indicators import INDICATORS as pkg_indicators
        from kaufman_indicators import get_indicator
        assert pkg_indicators is INDICATORS
        assert get_indicator("rsi") is INDICATORS["rsi"]
