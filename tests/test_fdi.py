"""Tests for Fractal Dimension Index (FDI)."""

import numpy as np
import pytest

from kaufman_indicators.market_quality.fdi import fdi


class TestFDI:
    def test_returns_correct_length(self):
        prices = np.linspace(100, 150, 60)
        result = fdi(prices, period=30)
        assert len(result) == 60

    def test_first_period_minus_one_are_nan(self):
        prices = np.linspace(100, 150, 60)
        result = fdi(prices, period=30)
        assert np.all(np.isnan(result[:29]))

    def test_trending_market_low_fdi(self):
        # A pure linear trend should have FDI closer to 1.0
        prices = np.linspace(100, 200, 100)
        result = fdi(prices, period=30)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.mean(valid) < 1.5

    def test_noisy_market_higher_fdi(self):
        # Random noise should have FDI closer to 1.5 or above
        rng = np.random.default_rng(42)
        prices = 100 + rng.standard_normal(200) * 5
        result = fdi(prices, period=30)
        valid = result[~np.isnan(result)]
        assert np.mean(valid) > 1.3

    def test_fdi_bounded(self):
        # FDI should be between 1.0 and 2.0
        rng = np.random.default_rng(7)
        prices = 100 + np.cumsum(rng.standard_normal(200))
        result = fdi(prices, period=30)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.9)  # allow small margin
        assert np.all(valid <= 2.1)

    def test_constant_prices_nan(self):
        # Constant prices → zero range → NaN
        prices = np.full(50, 100.0)
        result = fdi(prices, period=20)
        valid_idx = ~np.isnan(result[:19])  # warmup should be NaN
        assert not np.any(valid_idx)

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0, 102.0])
        result = fdi(prices, period=30)
        assert np.all(np.isnan(result))
