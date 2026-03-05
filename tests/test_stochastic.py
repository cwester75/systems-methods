"""Tests for the Stochastic Oscillator."""

import numpy as np
import pytest

from kaufman_indicators.momentum.stochastic import stochastic, StochasticResult


class TestStochastic:
    def test_returns_named_tuple(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = stochastic(high, low, close)
        assert isinstance(result, StochasticResult)

    def test_arrays_same_length_as_input(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = stochastic(high, low, close, k_period=14, d_period=3)
        assert len(result.k) == n
        assert len(result.d) == n

    def test_k_nan_for_warmup(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = stochastic(high, low, close, k_period=14)
        assert np.all(np.isnan(result.k[:13]))

    def test_d_nan_longer_than_k(self):
        n = 30
        high = np.linspace(105, 155, n)
        low = np.linspace(95, 145, n)
        close = np.linspace(100, 150, n)
        result = stochastic(high, low, close, k_period=14, d_period=3)
        # %D needs k_period + d_period - 2 bars of warmup
        assert np.all(np.isnan(result.d[:15]))

    def test_k_bounded_zero_to_100(self):
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.standard_normal(100))
        spread = np.abs(rng.standard_normal(100)) * 2
        high = base + spread
        low = base - spread
        close = base + rng.standard_normal(100) * 0.5
        # Clamp close within [low, high]
        close = np.clip(close, low, high)
        result = stochastic(high, low, close, k_period=14)
        valid_k = result.k[~np.isnan(result.k)]
        assert np.all(valid_k >= -1e-10)
        assert np.all(valid_k <= 100.0 + 1e-10)

    def test_close_at_high_gives_100(self):
        # When close == high for the entire window, %K should be 100
        n = 20
        high = np.arange(1.0, n + 1.0)
        low = high - 5.0
        close = high.copy()  # close at the high
        result = stochastic(high, low, close, k_period=5, d_period=3)
        # At index 4+, close == highest_high, so %K = 100
        valid_k = result.k[~np.isnan(result.k)]
        np.testing.assert_allclose(valid_k, 100.0, atol=1e-10)

    def test_close_at_low_gives_zero(self):
        # When close == lowest_low in each window, %K should be 0
        # Use flat low so lowest_low == low[i] == close[i]
        n = 20
        high = np.full(n, 110.0)
        low = np.full(n, 100.0)
        close = np.full(n, 100.0)  # close at the constant low
        result = stochastic(high, low, close, k_period=5, d_period=3)
        valid_k = result.k[~np.isnan(result.k)]
        np.testing.assert_allclose(valid_k, 0.0, atol=1e-10)

    def test_d_is_sma_of_k(self):
        rng = np.random.default_rng(7)
        n = 50
        base = 100 + np.cumsum(rng.standard_normal(n))
        high = base + 2.0
        low = base - 2.0
        close = base
        k_period, d_period = 14, 3
        result = stochastic(high, low, close, k_period=k_period, d_period=d_period)
        # Verify %D is the 3-period SMA of %K
        for i in range(k_period + d_period - 2, n):
            if np.isnan(result.d[i]):
                continue
            expected_d = np.mean(result.k[i - d_period + 1: i + 1])
            np.testing.assert_allclose(result.d[i], expected_d, atol=1e-10)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            stochastic(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))

    def test_too_short_input(self):
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5])
        result = stochastic(high, low, close, k_period=14)
        assert np.all(np.isnan(result.k))
        assert np.all(np.isnan(result.d))
