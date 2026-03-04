"""Tests for the Kaufman Adaptive Moving Average (KAMA)."""

import numpy as np
import pytest

from kaufman_indicators.trend.kama import kama
from kaufman_indicators.trend.efficiency_ratio import efficiency_ratio


class TestEfficiencyRatio:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 21.0)
        er = efficiency_ratio(prices, period=10)
        assert len(er) == len(prices)

    def test_first_period_values_are_nan(self):
        prices = np.arange(1.0, 21.0)
        er = efficiency_ratio(prices, period=10)
        assert np.all(np.isnan(er[:10]))

    def test_trending_market_er_near_one(self):
        # Perfectly linear uptrend → ER should be 1.0
        prices = np.linspace(100.0, 200.0, 50)
        er = efficiency_ratio(prices, period=10)
        valid = er[~np.isnan(er)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-6)

    def test_sideways_market_er_near_zero(self):
        # Alternating ±1 → net change ≈ 0, ER should be near 0
        prices = np.array([100.0 + ((-1) ** i) for i in range(30)])
        er = efficiency_ratio(prices, period=10)
        valid = er[~np.isnan(er)]
        assert np.all(valid < 0.2)

    def test_values_bounded_zero_to_one(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(100))
        er = efficiency_ratio(prices, period=10)
        valid = er[~np.isnan(er)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0, 3.0])
        er = efficiency_ratio(prices, period=10)
        assert np.all(np.isnan(er))


class TestKAMA:
    def test_returns_correct_length(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = kama(prices)
        assert len(result) == 50

    def test_first_period_values_are_nan(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = kama(prices, period=10)
        assert np.all(np.isnan(result[:10]))

    def test_seed_value(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = kama(prices, period=10)
        # KAMA is seeded at prices[period]
        assert result[10] == prices[10]

    def test_trending_kama_follows_price(self):
        # In a strong trend KAMA should be close to the price
        prices = np.linspace(100.0, 200.0, 100)
        result = kama(prices, period=10)
        valid_kama = result[~np.isnan(result)]
        valid_prices = prices[~np.isnan(result)]
        # Relative error should be small for a perfect trend
        rel_error = np.abs(valid_kama - valid_prices) / valid_prices
        assert np.all(rel_error < 0.15)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = kama(prices, period=10)
        assert np.all(np.isnan(result))

    def test_custom_fast_slow(self):
        prices = np.linspace(100.0, 200.0, 60)
        result = kama(prices, period=10, fast=3, slow=20)
        assert not np.all(np.isnan(result))

    def test_no_nan_after_seed(self):
        prices = np.linspace(100.0, 200.0, 50)
        result = kama(prices, period=10)
        # After the seed index there should be no NaN
        assert not np.any(np.isnan(result[10:]))
