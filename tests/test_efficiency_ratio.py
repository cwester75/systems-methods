"""Tests for the Efficiency Ratio (ER) indicator."""

import numpy as np
import pytest

from kaufman_indicators.trend.efficiency_ratio import efficiency_ratio


class TestEfficiencyRatio:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 31.0)
        result = efficiency_ratio(prices, period=10)
        assert len(result) == 30

    def test_first_period_values_are_nan(self):
        prices = np.arange(1.0, 31.0)
        result = efficiency_ratio(prices, period=10)
        assert np.all(np.isnan(result[:10]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 31.0)
        result = efficiency_ratio(prices, period=10)
        assert not np.any(np.isnan(result[10:]))

    def test_straight_trend_er_is_one(self):
        # Perfectly linear uptrend: net change == sum of abs changes → ER = 1
        prices = np.linspace(100.0, 200.0, 30)
        result = efficiency_ratio(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_straight_downtrend_er_is_one(self):
        # Perfectly linear downtrend: |net change| == sum of abs changes → ER = 1
        prices = np.linspace(200.0, 100.0, 30)
        result = efficiency_ratio(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_constant_prices_er_is_zero(self):
        # No net change, no daily changes → 0/0 handled as 0
        prices = np.full(30, 100.0)
        result = efficiency_ratio(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_er_between_zero_and_one(self):
        # Random walk should have ER between 0 and 1
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(100))
        result = efficiency_ratio(prices, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_known_value(self):
        # Manual: prices = [10, 12, 11, 13, 15]
        # period=4: net = |15 - 10| = 5
        # abs changes = |2| + |1| + |2| + |2| = 7
        # ER = 5/7
        prices = np.array([10.0, 12.0, 11.0, 13.0, 15.0])
        result = efficiency_ratio(prices, period=4)
        np.testing.assert_allclose(result[4], 5.0 / 7.0, atol=1e-10)

    def test_noisy_series_lower_er(self):
        # A noisy series should have lower ER than a smooth trend
        smooth = np.linspace(100, 150, 40)
        noisy = smooth.copy()
        noisy[1::2] += 5.0  # add zigzag
        noisy[2::2] -= 5.0
        er_smooth = efficiency_ratio(smooth, period=10)
        er_noisy = efficiency_ratio(noisy, period=10)
        assert np.nanmean(er_smooth) > np.nanmean(er_noisy)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = efficiency_ratio(prices, period=5)
        assert np.all(np.isnan(result))

    def test_default_period(self):
        prices = np.linspace(100.0, 200.0, 30)
        result = efficiency_ratio(prices)  # default period=10
        assert np.all(np.isnan(result[:10]))
        assert not np.any(np.isnan(result[10:]))

    def test_period_equals_length(self):
        # n == period → should return all NaN
        prices = np.arange(1.0, 11.0)
        result = efficiency_ratio(prices, period=10)
        assert np.all(np.isnan(result))
