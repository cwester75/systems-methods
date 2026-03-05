"""Tests for Realized Volatility and Rolling Standard Deviation."""

import numpy as np
import pytest

from kaufman_indicators.volatility.realized_vol import realized_vol
from kaufman_indicators.utils.rolling import rolling_std


class TestRollingStd:
    def test_returns_correct_length(self):
        a = np.arange(1.0, 21.0)
        result = rolling_std(a, window=5)
        assert len(result) == 20

    def test_first_window_minus_one_are_nan(self):
        a = np.arange(1.0, 21.0)
        result = rolling_std(a, window=5)
        assert np.all(np.isnan(result[:4]))

    def test_no_nan_after_warmup(self):
        a = np.arange(1.0, 21.0)
        result = rolling_std(a, window=5)
        assert not np.any(np.isnan(result[4:]))

    def test_known_value(self):
        a = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        result = rolling_std(a, window=4, ddof=1)
        # First valid at index 3: std([2, 4, 4, 4]) = sqrt(((−1)²+1²+1²+1²) / 3) ≈ 1.1547
        expected = np.std([2.0, 4.0, 4.0, 4.0], ddof=1)
        np.testing.assert_allclose(result[3], expected, atol=1e-10)

    def test_constant_values_zero_std(self):
        a = np.full(20, 42.0)
        result = rolling_std(a, window=5, ddof=1)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_ddof_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_pop = rolling_std(a, window=3, ddof=0)
        result_sample = rolling_std(a, window=3, ddof=1)
        # Population std < sample std
        assert result_pop[2] < result_sample[2]

    def test_too_short_input(self):
        a = np.array([1.0, 2.0])
        result = rolling_std(a, window=5)
        assert np.all(np.isnan(result))


class TestRealizedVol:
    def test_returns_correct_length(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = realized_vol(prices, period=20)
        assert len(result) == 50

    def test_first_period_values_are_nan(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = realized_vol(prices, period=20)
        assert np.all(np.isnan(result[:20]))

    def test_no_nan_after_warmup(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = realized_vol(prices, period=20)
        assert not np.any(np.isnan(result[20:]))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
        result = realized_vol(prices, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_constant_prices_zero_vol(self):
        prices = np.full(40, 100.0)
        result = realized_vol(prices, period=20)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_annualization(self):
        rng = np.random.default_rng(7)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(60) * 0.01))
        annualized = realized_vol(prices, period=20, annualize=True)
        raw = realized_vol(prices, period=20, annualize=False)
        ratio = annualized[30] / raw[30]
        np.testing.assert_allclose(ratio, np.sqrt(252), atol=1e-10)

    def test_custom_periods_per_year(self):
        rng = np.random.default_rng(7)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(60) * 0.01))
        result_252 = realized_vol(prices, period=20, periods_per_year=252)
        result_365 = realized_vol(prices, period=20, periods_per_year=365)
        # Higher periods_per_year → higher annualized vol
        assert result_365[30] > result_252[30]

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0, 102.0])
        result = realized_vol(prices, period=20)
        assert np.all(np.isnan(result))
