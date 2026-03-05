"""Tests for Price Series Entropy."""

import numpy as np
import pytest

from kaufman_indicators.market_quality.entropy import price_entropy


class TestShannonEntropy:
    def test_returns_correct_length(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
        result = price_entropy(prices, period=50, method="shannon")
        assert len(result) == 100

    def test_first_period_values_are_nan(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
        result = price_entropy(prices, period=50, method="shannon")
        assert np.all(np.isnan(result[:50]))

    def test_has_valid_values_after_warmup(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
        result = price_entropy(prices, period=50, method="shannon")
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(100) * 0.01))
        result = price_entropy(prices, period=50, method="shannon")
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_diverse_returns_higher_entropy(self):
        # Wide-ranging returns → higher Shannon entropy
        rng = np.random.default_rng(42)
        prices_diverse = 100 * np.exp(np.cumsum(rng.standard_normal(150) * 0.03))
        # Nearly constant returns → low entropy (all fall in one bin)
        prices_steady = 100 * np.exp(np.cumsum(np.full(150, 0.001)))

        ent_diverse = price_entropy(prices_diverse, period=50, method="shannon", bins=10)
        ent_steady = price_entropy(prices_steady, period=50, method="shannon", bins=10)

        valid_d = ent_diverse[~np.isnan(ent_diverse)]
        valid_s = ent_steady[~np.isnan(ent_steady)]
        # Diverse returns should have higher entropy on average
        assert np.mean(valid_d) > np.mean(valid_s)

    def test_bounded_by_log2_bins(self):
        # Shannon entropy with n bins is bounded by log2(n)
        bins = 10
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(200) * 0.01))
        result = price_entropy(prices, period=50, method="shannon", bins=bins)
        valid = result[~np.isnan(result)]
        assert np.all(valid <= np.log2(bins) + 0.01)

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0])
        result = price_entropy(prices, period=50, method="shannon")
        assert np.all(np.isnan(result))


class TestApproximateEntropy:
    def test_returns_valid_values(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(150) * 0.01))
        result = price_entropy(prices, period=50, method="approximate")
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(150) * 0.01))
        result = price_entropy(prices, period=50, method="approximate")
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_random_higher_than_structured(self):
        # Random data → higher ApEn than structured (linear) data
        rng = np.random.default_rng(42)
        prices_random = 100 + np.cumsum(rng.standard_normal(200))
        prices_linear = np.linspace(100, 200, 200)

        apen_random = price_entropy(prices_random, period=50, method="approximate")
        apen_linear = price_entropy(prices_linear, period=50, method="approximate")

        valid_r = apen_random[~np.isnan(apen_random)]
        valid_l = apen_linear[~np.isnan(apen_linear)]
        if len(valid_r) > 0 and len(valid_l) > 0:
            assert np.mean(valid_r) > np.mean(valid_l)


class TestEntropySyntheticExpectations:
    def test_linear_ramp_low_entropy(self):
        # Constant log-returns → all in one bin → low entropy
        prices = np.exp(np.linspace(np.log(100), np.log(200), 150))
        result = price_entropy(prices, period=50, method="shannon", bins=10)
        valid = result[~np.isnan(result)]
        # Single-value log returns → most mass in 1-2 bins → entropy < 1
        assert np.mean(valid) < 2.0

    def test_noisy_series_high_entropy(self):
        rng = np.random.default_rng(42)
        prices = 100 * np.exp(np.cumsum(rng.standard_normal(150) * 0.05))
        result = price_entropy(prices, period=50, method="shannon", bins=10)
        valid = result[~np.isnan(result)]
        # Noisy returns spread across bins → high entropy
        assert np.mean(valid) > 2.0


class TestEntropyValidation:
    def test_invalid_method_raises(self):
        prices = np.linspace(100, 200, 100)
        with pytest.raises(ValueError):
            price_entropy(prices, method="invalid")
