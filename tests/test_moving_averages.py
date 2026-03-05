"""Tests for Simple Moving Average (SMA) and Exponential Moving Average (EMA)."""

import numpy as np
import pytest

from kaufman_indicators.trend.moving_averages import sma, ema


class TestSMA:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 21.0)
        result = sma(prices, period=5)
        assert len(result) == 20

    def test_first_period_minus_one_are_nan(self):
        prices = np.arange(1.0, 21.0)
        result = sma(prices, period=5)
        assert np.all(np.isnan(result[:4]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 21.0)
        result = sma(prices, period=5)
        assert not np.any(np.isnan(result[4:]))

    def test_known_values(self):
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = sma(prices, period=3)
        # SMA(3) at index 2 = (1+2+3)/3 = 2.0
        np.testing.assert_allclose(result[2], 2.0, atol=1e-10)
        # SMA(3) at index 3 = (2+3+4)/3 = 3.0
        np.testing.assert_allclose(result[3], 3.0, atol=1e-10)
        # SMA(3) at index 6 = (5+6+7)/3 = 6.0
        np.testing.assert_allclose(result[6], 6.0, atol=1e-10)

    def test_constant_prices(self):
        prices = np.full(20, 42.0)
        result = sma(prices, period=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 42.0, atol=1e-10)

    def test_period_one(self):
        prices = np.array([10.0, 20.0, 30.0])
        result = sma(prices, period=1)
        np.testing.assert_allclose(result, prices, atol=1e-10)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = sma(prices, period=5)
        assert np.all(np.isnan(result))


class TestEMA:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 21.0)
        result = ema(prices, period=5)
        assert len(result) == 20

    def test_first_period_minus_one_are_nan(self):
        prices = np.arange(1.0, 21.0)
        result = ema(prices, period=5)
        assert np.all(np.isnan(result[:4]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 21.0)
        result = ema(prices, period=5)
        assert not np.any(np.isnan(result[4:]))

    def test_seed_is_sma(self):
        prices = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        result = ema(prices, period=5)
        expected_seed = np.mean(prices[:5])  # (2+4+6+8+10)/5 = 6.0
        np.testing.assert_allclose(result[4], expected_seed, atol=1e-10)

    def test_known_recursive_step(self):
        prices = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        period = 5
        alpha = 2.0 / (period + 1)
        result = ema(prices, period=period)
        seed = np.mean(prices[:5])
        expected_next = seed + alpha * (prices[5] - seed)
        np.testing.assert_allclose(result[5], expected_next, atol=1e-10)

    def test_constant_prices(self):
        prices = np.full(30, 100.0)
        result = ema(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 100.0, atol=1e-10)

    def test_custom_alpha(self):
        prices = np.arange(1.0, 11.0)
        result_default = ema(prices, period=5)
        result_custom = ema(prices, period=5, alpha=0.5)
        # Different alpha should produce different results
        assert not np.allclose(result_default[4:], result_custom[4:])

    def test_ema_follows_trend(self):
        # EMA should lag behind a linear uptrend
        prices = np.linspace(100.0, 200.0, 50)
        result = ema(prices, period=10)
        # After warmup, EMA should be below the price in an uptrend
        for i in range(15, 50):
            assert result[i] < prices[i]

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = ema(prices, period=5)
        assert np.all(np.isnan(result))
