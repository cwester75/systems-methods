"""Tests for Williams %R."""

import numpy as np
import pytest

from kaufman_indicators.range.williams_r import williams_r


class TestWilliamsR:
    def test_returns_correct_length(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = williams_r(high, low, close, period=14)
        assert len(result) == n

    def test_first_period_minus_one_are_nan(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = williams_r(high, low, close, period=14)
        assert np.all(np.isnan(result[:13]))

    def test_bounded_minus_100_to_zero(self):
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.standard_normal(80))
        spread = np.abs(rng.standard_normal(80)) * 2 + 0.5
        high = base + spread
        low = base - spread
        close = np.clip(base, low, high)
        result = williams_r(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100.0 - 1e-10)
        assert np.all(valid <= 0.0 + 1e-10)

    def test_close_at_high_gives_zero(self):
        # When close == highest high, %R = 0
        n = 20
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 110.0)  # close at the constant high
        result = williams_r(high, low, close, period=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_close_at_low_gives_minus_100(self):
        # When close == lowest low, %R = -100
        n = 20
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 90.0)  # close at the constant low
        result = williams_r(high, low, close, period=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, -100.0, atol=1e-10)

    def test_close_at_midpoint_gives_minus_50(self):
        n = 20
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        close = np.full(n, 100.0)  # midpoint
        result = williams_r(high, low, close, period=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, -50.0, atol=1e-10)

    def test_relationship_to_stochastic(self):
        # %R = %K - 100 (when using same period and same inputs)
        from kaufman_indicators.momentum.stochastic import stochastic
        rng = np.random.default_rng(7)
        n = 50
        base = 100 + np.cumsum(rng.standard_normal(n))
        high = base + 2.0
        low = base - 2.0
        close = base
        period = 14
        wr = williams_r(high, low, close, period=period)
        stoch = stochastic(high, low, close, k_period=period)
        # %R = -(100 - %K) = %K - 100
        for i in range(period - 1, n):
            if np.isnan(wr[i]) or np.isnan(stoch.k[i]):
                continue
            np.testing.assert_allclose(wr[i], stoch.k[i] - 100.0, atol=1e-10)

    def test_monotonic_uptrend_near_zero(self):
        # In an uptrend, close near highest high → %R near 0
        n = 40
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0
        result = williams_r(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.mean(valid[-10:]) > -30  # near 0, not -100

    def test_monotonic_downtrend_near_minus_100(self):
        # In a downtrend, close near lowest low → %R near -100
        n = 40
        close = np.linspace(200, 100, n)
        high = close + 2.0
        low = close - 2.0
        result = williams_r(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.mean(valid[-10:]) < -70  # near -100

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            williams_r(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))

    def test_too_short_input(self):
        high = np.array([110.0, 112.0])
        low = np.array([90.0, 88.0])
        close = np.array([100.0, 100.0])
        result = williams_r(high, low, close, period=14)
        assert np.all(np.isnan(result))
