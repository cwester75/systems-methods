"""Tests for Donchian Channels."""

import numpy as np
import pytest

from kaufman_indicators.range.donchian import donchian_channels, DonchianResult


class TestDonchianChannels:
    def test_returns_named_tuple(self):
        high = np.linspace(105, 155, 30)
        low = np.linspace(95, 145, 30)
        result = donchian_channels(high, low)
        assert isinstance(result, DonchianResult)

    def test_arrays_same_length_as_input(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        result = donchian_channels(high, low, period=20)
        assert len(result.upper) == n
        assert len(result.lower) == n
        assert len(result.mid) == n

    def test_first_period_minus_one_are_nan(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        result = donchian_channels(high, low, period=20)
        assert np.all(np.isnan(result.upper[:19]))
        assert np.all(np.isnan(result.lower[:19]))
        assert np.all(np.isnan(result.mid[:19]))

    def test_upper_is_rolling_max_of_high(self):
        rng = np.random.default_rng(42)
        high = 100 + np.cumsum(rng.standard_normal(40)) + 2
        low = high - 4
        period = 10
        result = donchian_channels(high, low, period=period)
        for i in range(period - 1, len(high)):
            expected = np.max(high[i - period + 1: i + 1])
            np.testing.assert_allclose(result.upper[i], expected, atol=1e-10)

    def test_lower_is_rolling_min_of_low(self):
        rng = np.random.default_rng(42)
        high = 100 + np.cumsum(rng.standard_normal(40)) + 2
        low = high - 4
        period = 10
        result = donchian_channels(high, low, period=period)
        for i in range(period - 1, len(low)):
            expected = np.min(low[i - period + 1: i + 1])
            np.testing.assert_allclose(result.lower[i], expected, atol=1e-10)

    def test_mid_is_average_of_upper_and_lower(self):
        rng = np.random.default_rng(7)
        high = 100 + np.cumsum(rng.standard_normal(40)) + 2
        low = high - 4
        result = donchian_channels(high, low, period=10)
        for i in range(9, len(high)):
            expected = (result.upper[i] + result.lower[i]) / 2.0
            np.testing.assert_allclose(result.mid[i], expected, atol=1e-10)

    def test_upper_ge_lower(self):
        rng = np.random.default_rng(42)
        high = 100 + np.cumsum(rng.standard_normal(50)) + 3
        low = high - 6
        result = donchian_channels(high, low, period=14)
        for i in range(13, 50):
            assert result.upper[i] >= result.lower[i]

    def test_constant_range(self):
        n = 30
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        result = donchian_channels(high, low, period=10)
        valid_upper = result.upper[~np.isnan(result.upper)]
        valid_lower = result.lower[~np.isnan(result.lower)]
        valid_mid = result.mid[~np.isnan(result.mid)]
        np.testing.assert_allclose(valid_upper, 110.0, atol=1e-10)
        np.testing.assert_allclose(valid_lower, 90.0, atol=1e-10)
        np.testing.assert_allclose(valid_mid, 100.0, atol=1e-10)

    def test_monotonic_uptrend_widening_channel(self):
        # In an uptrend, upper tracks the latest high, lower lags → channel widens
        n = 40
        high = np.linspace(105, 205, n)
        low = np.linspace(95, 195, n)
        period = 10
        result = donchian_channels(high, low, period=period)
        # Channel width should be constant for a linear series
        for i in range(period - 1, n):
            width = result.upper[i] - result.lower[i]
            expected_width = high[i] - low[i - period + 1]
            np.testing.assert_allclose(width, expected_width, atol=1e-10)

    def test_flat_series_zero_width(self):
        # Flat H and L → upper == lower → width = 0
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        result = donchian_channels(high, low, period=10)
        valid_width = result.upper[9:] - result.lower[9:]
        np.testing.assert_allclose(valid_width, 0.0, atol=1e-10)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            donchian_channels(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_short_input(self):
        high = np.array([110.0, 112.0])
        low = np.array([90.0, 88.0])
        result = donchian_channels(high, low, period=20)
        assert np.all(np.isnan(result.upper))
        assert np.all(np.isnan(result.lower))
        assert np.all(np.isnan(result.mid))
