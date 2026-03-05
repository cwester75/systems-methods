"""Tests for the Rate of Change (ROC) indicator."""

import numpy as np
import pytest

from kaufman_indicators.momentum.roc import roc


class TestROC:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 31.0)
        result = roc(prices, period=12)
        assert len(result) == 30

    def test_first_period_values_are_nan(self):
        prices = np.arange(1.0, 31.0)
        result = roc(prices, period=12)
        assert np.all(np.isnan(result[:12]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 31.0)
        result = roc(prices, period=12)
        assert not np.any(np.isnan(result[12:]))

    def test_known_values(self):
        # price[0]=100, price[5]=150 → ROC = (150-100)/100 * 100 = 50%
        prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
        result = roc(prices, period=5)
        np.testing.assert_allclose(result[5], 50.0, atol=1e-10)

    def test_constant_prices_zero_roc(self):
        prices = np.full(30, 50.0)
        result = roc(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_uptrend_positive(self):
        prices = np.linspace(100.0, 200.0, 40)
        result = roc(prices, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_downtrend_negative(self):
        prices = np.linspace(200.0, 100.0, 40)
        result = roc(prices, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)

    def test_zero_price_handled(self):
        # If a prior price is 0, ROC should be NaN (division by zero)
        prices = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        result = roc(prices, period=3)
        assert np.isnan(result[3])  # 1.0 vs 0.0 → NaN

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = roc(prices, period=5)
        assert np.all(np.isnan(result))

    def test_relationship_to_momentum(self):
        # ROC = momentum / price_{t-n} * 100
        prices = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 18.0, 13.0])
        period = 3
        result = roc(prices, period=period)
        for i in range(period, len(prices)):
            expected = (prices[i] - prices[i - period]) / prices[i - period] * 100.0
            np.testing.assert_allclose(result[i], expected, atol=1e-10)
