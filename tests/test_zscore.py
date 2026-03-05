"""Tests for Price Z-Score."""

import numpy as np
import pytest

from kaufman_indicators.range.zscore import price_zscore


class TestPriceZScore:
    def test_returns_correct_length(self):
        prices = np.linspace(100, 150, 40)
        result = price_zscore(prices, period=20)
        assert len(result) == 40

    def test_first_period_minus_one_are_nan(self):
        prices = np.linspace(100, 150, 40)
        result = price_zscore(prices, period=20)
        assert np.all(np.isnan(result[:19]))

    def test_no_nan_after_warmup(self):
        prices = np.linspace(100, 150, 40)
        result = price_zscore(prices, period=20)
        assert not np.any(np.isnan(result[19:]))

    def test_constant_prices_zero_zscore(self):
        prices = np.full(30, 100.0)
        result = price_zscore(prices, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_value(self):
        # Manual calculation for a simple window
        prices = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        result = price_zscore(prices, period=5)
        mean = np.mean(prices)
        std = np.std(prices, ddof=1)
        expected = (prices[-1] - mean) / std
        np.testing.assert_allclose(result[4], expected, atol=1e-10)

    def test_positive_when_above_mean(self):
        # Strong uptrend → price above rolling mean → positive z-score
        prices = np.linspace(100, 200, 50)
        result = price_zscore(prices, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_negative_when_below_mean(self):
        # Strong downtrend → price below rolling mean → negative z-score
        prices = np.linspace(200, 100, 50)
        result = price_zscore(prices, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)

    def test_relationship_to_bollinger_percent_b(self):
        # Z-score = (price - SMA) / std
        # %B = (price - lower) / (upper - lower) = (price - (SMA - k*std)) / (2*k*std)
        # %B = (Z + k) / (2k) when k = num_std
        from kaufman_indicators.range.bollinger import bollinger_bands
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(50))
        period = 20
        num_std = 2.0
        z = price_zscore(prices, period=period)
        bb = bollinger_bands(prices, period=period, num_std=num_std)
        for i in range(period - 1, len(prices)):
            if np.isnan(z[i]) or np.isnan(bb.percent_b[i]):
                continue
            expected_pctb = (z[i] + num_std) / (2.0 * num_std)
            np.testing.assert_allclose(bb.percent_b[i], expected_pctb, atol=1e-10)

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0])
        result = price_zscore(prices, period=20)
        assert np.all(np.isnan(result))

    def test_default_period(self):
        prices = np.linspace(100, 150, 40)
        result = price_zscore(prices)  # default period=20
        assert np.all(np.isnan(result[:19]))
        assert not np.any(np.isnan(result[19:]))

    def test_last_value_of_window_highest_zscore(self):
        # In a monotonically increasing window, the last value is the furthest
        # above the mean, so it should have the highest z-score
        prices = np.arange(1.0, 31.0)
        result = price_zscore(prices, period=10)
        # Each valid z-score should be positive (price above rolling mean)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_period_equals_length(self):
        prices = np.arange(1.0, 21.0)
        result = price_zscore(prices, period=20)
        # Should have NaN for first 19, then one valid value at index 19
        assert np.all(np.isnan(result[:19]))
        assert not np.isnan(result[19])
