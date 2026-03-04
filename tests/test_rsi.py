"""Tests for the Relative Strength Index (RSI)."""

import numpy as np
import pytest

from kaufman_indicators.momentum.rsi import rsi


class TestRSI:
    def test_returns_correct_length(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = rsi(prices)
        assert len(result) == 50

    def test_first_period_values_are_nan(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = rsi(prices, period=14)
        assert np.all(np.isnan(result[:14]))

    def test_values_bounded_zero_to_100(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(200))
        result = rsi(prices, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_all_gains_gives_100(self):
        # Strictly increasing prices → RSI should be 100 after initial period
        prices = np.linspace(100.0, 200.0, 30)
        result = rsi(prices, period=14)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 100.0, atol=1e-6)

    def test_all_losses_gives_zero(self):
        # Strictly decreasing prices → RSI should approach 0
        prices = np.linspace(200.0, 100.0, 30)
        result = rsi(prices, period=14)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-6)

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0])
        result = rsi(prices, period=14)
        assert np.all(np.isnan(result))

    def test_custom_period(self):
        prices = np.linspace(100.0, 150.0, 30)
        result = rsi(prices, period=7)
        assert len(result) == 30
        assert np.all(np.isnan(result[:7]))
        assert not np.any(np.isnan(result[7:]))

    def test_rsi_midpoint_neutral_market(self):
        # Alternating +1/-1 changes → RSI should hover near 50
        rng = np.random.default_rng(0)
        prices = np.array([100.0])
        for _ in range(200):
            prices = np.append(prices, prices[-1] + rng.choice([-1.0, 1.0]))
        result = rsi(prices, period=14)
        valid = result[~np.isnan(result)]
        assert 30 < np.mean(valid) < 70
