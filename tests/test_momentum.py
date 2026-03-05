"""Tests for the Momentum indicator."""

import numpy as np
import pytest

from kaufman_indicators.momentum.momentum import momentum


class TestMomentum:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 31.0)
        result = momentum(prices, period=10)
        assert len(result) == 30

    def test_first_period_values_are_nan(self):
        prices = np.arange(1.0, 31.0)
        result = momentum(prices, period=10)
        assert np.all(np.isnan(result[:10]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 31.0)
        result = momentum(prices, period=10)
        assert not np.any(np.isnan(result[10:]))

    def test_known_values(self):
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = momentum(prices, period=3)
        # momentum[3] = 40 - 10 = 30
        np.testing.assert_allclose(result[3], 30.0, atol=1e-10)
        # momentum[4] = 50 - 20 = 30
        np.testing.assert_allclose(result[4], 30.0, atol=1e-10)
        # momentum[5] = 60 - 30 = 30
        np.testing.assert_allclose(result[5], 30.0, atol=1e-10)

    def test_constant_prices_zero_momentum(self):
        prices = np.full(30, 42.0)
        result = momentum(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_uptrend_positive(self):
        prices = np.linspace(100.0, 200.0, 40)
        result = momentum(prices, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_downtrend_negative(self):
        prices = np.linspace(200.0, 100.0, 40)
        result = momentum(prices, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)

    def test_relationship_to_roc(self):
        # momentum = ROC / 100 * price_{t-n}
        from kaufman_indicators.momentum.roc import roc
        prices = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 18.0, 13.0, 16.0])
        period = 3
        mom = momentum(prices, period=period)
        r = roc(prices, period=period)
        for i in range(period, len(prices)):
            expected = r[i] / 100.0 * prices[i - period]
            np.testing.assert_allclose(mom[i], expected, atol=1e-10)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0])
        result = momentum(prices, period=5)
        assert np.all(np.isnan(result))

    def test_default_period(self):
        prices = np.arange(1.0, 21.0)
        result = momentum(prices)  # default period=10
        assert np.all(np.isnan(result[:10]))
        np.testing.assert_allclose(result[10], 10.0, atol=1e-10)

    def test_period_equals_length(self):
        prices = np.arange(1.0, 11.0)
        result = momentum(prices, period=10)
        assert np.all(np.isnan(result))

    def test_alternating_prices(self):
        # Alternating prices with period=2 → momentum should alternate
        prices = np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
        result = momentum(prices, period=2)
        np.testing.assert_allclose(result[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[3], 0.0, atol=1e-10)

    def test_single_step_change(self):
        # Flat, then jump, then flat → momentum reflects the jump
        prices = np.array([100.0] * 5 + [110.0] * 5)
        result = momentum(prices, period=3)
        # At index 5: 110 - 100 = 10
        np.testing.assert_allclose(result[5], 10.0, atol=1e-10)
        # At index 8: 110 - 110 = 0 (both sides past the jump)
        np.testing.assert_allclose(result[8], 0.0, atol=1e-10)
