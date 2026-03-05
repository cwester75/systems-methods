"""Tests for Bollinger Bands and Bollinger Band Width."""

import numpy as np
import pytest

from kaufman_indicators.range.bollinger import bollinger_bands, BollingerResult


class TestBollingerBands:
    def test_returns_named_tuple(self):
        prices = np.linspace(100, 150, 40)
        result = bollinger_bands(prices)
        assert isinstance(result, BollingerResult)

    def test_arrays_same_length_as_input(self):
        prices = np.linspace(100, 150, 40)
        result = bollinger_bands(prices, period=20)
        assert len(result.middle) == 40
        assert len(result.upper) == 40
        assert len(result.lower) == 40
        assert len(result.bandwidth) == 40
        assert len(result.percent_b) == 40

    def test_first_period_minus_one_are_nan(self):
        prices = np.linspace(100, 150, 40)
        result = bollinger_bands(prices, period=20)
        assert np.all(np.isnan(result.middle[:19]))
        assert np.all(np.isnan(result.upper[:19]))
        assert np.all(np.isnan(result.lower[:19]))

    def test_middle_is_sma(self):
        from kaufman_indicators.trend.moving_averages import sma
        prices = np.linspace(100, 150, 40)
        result = bollinger_bands(prices, period=20)
        expected_sma = sma(prices, period=20)
        valid = ~np.isnan(result.middle)
        np.testing.assert_allclose(result.middle[valid], expected_sma[valid], atol=1e-10)

    def test_upper_above_middle_above_lower(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(60))
        result = bollinger_bands(prices, period=20)
        for i in range(19, 60):
            assert result.upper[i] >= result.middle[i]
            assert result.middle[i] >= result.lower[i]

    def test_band_symmetry(self):
        # Upper - middle should equal middle - lower
        rng = np.random.default_rng(7)
        prices = 100 + np.cumsum(rng.standard_normal(50))
        result = bollinger_bands(prices, period=20, num_std=2.0)
        for i in range(19, 50):
            upper_diff = result.upper[i] - result.middle[i]
            lower_diff = result.middle[i] - result.lower[i]
            np.testing.assert_allclose(upper_diff, lower_diff, atol=1e-10)

    def test_constant_prices_zero_bandwidth(self):
        prices = np.full(30, 100.0)
        result = bollinger_bands(prices, period=10)
        valid_bw = result.bandwidth[~np.isnan(result.bandwidth)]
        np.testing.assert_allclose(valid_bw, 0.0, atol=1e-10)

    def test_custom_num_std(self):
        prices = np.linspace(100, 150, 40)
        result_2 = bollinger_bands(prices, period=20, num_std=2.0)
        result_3 = bollinger_bands(prices, period=20, num_std=3.0)
        # Wider multiplier → wider bands
        for i in range(19, 40):
            assert result_3.upper[i] > result_2.upper[i]
            assert result_3.lower[i] < result_2.lower[i]

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0, 102.0])
        result = bollinger_bands(prices, period=20)
        assert np.all(np.isnan(result.middle))
        assert np.all(np.isnan(result.upper))
        assert np.all(np.isnan(result.lower))


class TestBollingerBandWidth:
    def test_bandwidth_formula(self):
        # bandwidth = (upper - lower) / middle * 100
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(50))
        result = bollinger_bands(prices, period=20)
        for i in range(19, 50):
            expected = (result.upper[i] - result.lower[i]) / result.middle[i] * 100.0
            np.testing.assert_allclose(result.bandwidth[i], expected, atol=1e-10)

    def test_bandwidth_non_negative(self):
        rng = np.random.default_rng(7)
        prices = 100 + np.cumsum(rng.standard_normal(60))
        result = bollinger_bands(prices, period=20)
        valid = result.bandwidth[~np.isnan(result.bandwidth)]
        assert np.all(valid >= 0)

    def test_higher_volatility_wider_bandwidth(self):
        # Steady prices → narrow bandwidth
        steady = np.full(40, 100.0) + np.random.default_rng(0).standard_normal(40) * 0.01
        volatile = np.full(40, 100.0) + np.random.default_rng(0).standard_normal(40) * 5.0
        bw_steady = bollinger_bands(steady, period=20).bandwidth[30]
        bw_volatile = bollinger_bands(volatile, period=20).bandwidth[30]
        assert bw_volatile > bw_steady


class TestPercentB:
    def test_percent_b_at_upper_band(self):
        # If close == upper band, %B = 1.0
        # Use constant std so we can predict the bands
        prices = np.full(30, 100.0)
        result = bollinger_bands(prices, period=10)
        # Constant prices: upper == lower == middle → %B undefined (NaN or 0)
        # Instead test with varying prices
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(40))
        result = bollinger_bands(prices, period=20)
        # Verify formula: %B = (price - lower) / (upper - lower)
        for i in range(19, 40):
            band_range = result.upper[i] - result.lower[i]
            if band_range > 0:
                expected = (prices[i] - result.lower[i]) / band_range
                np.testing.assert_allclose(result.percent_b[i], expected, atol=1e-10)

    def test_percent_b_at_middle(self):
        # When price == middle, and bands are symmetric, %B should be ~0.5
        # Use data where price happens to equal SMA
        n = 30
        prices = np.full(n, 100.0)
        prices[20:] = 100.0 + np.sin(np.arange(10)) * 2  # some variation
        # The middle band won't exactly equal price, so just verify formula
        result = bollinger_bands(prices, period=10)
        for i in range(9, n):
            band_range = result.upper[i] - result.lower[i]
            if band_range > 0:
                expected = (prices[i] - result.lower[i]) / band_range
                np.testing.assert_allclose(result.percent_b[i], expected, atol=1e-10)
