"""Tests for the MACD indicator."""

import numpy as np
import pytest

from kaufman_indicators.momentum.macd import macd, MACDResult


class TestMACD:
    def test_returns_named_tuple(self):
        prices = np.linspace(100.0, 150.0, 60)
        result = macd(prices)
        assert isinstance(result, MACDResult)

    def test_arrays_same_length_as_input(self):
        prices = np.linspace(100.0, 150.0, 60)
        result = macd(prices)
        assert len(result.macd_line) == 60
        assert len(result.signal) == 60
        assert len(result.histogram) == 60

    def test_nan_where_insufficient_data(self):
        prices = np.linspace(100.0, 150.0, 60)
        result = macd(prices, fast=12, slow=26, signal_period=9)
        # Need at least 26 bars for slow EMA, then 9 more for signal
        assert np.all(np.isnan(result.macd_line[:25]))

    def test_histogram_equals_macd_minus_signal(self):
        rng = np.random.default_rng(7)
        prices = 100 + np.cumsum(rng.standard_normal(100))
        result = macd(prices)
        valid = ~(np.isnan(result.macd_line) | np.isnan(result.signal))
        np.testing.assert_allclose(
            result.histogram[valid],
            result.macd_line[valid] - result.signal[valid],
            atol=1e-10,
        )

    def test_uptrend_positive_macd(self):
        # Strong uptrend: fast EMA > slow EMA → MACD line should be positive
        prices = np.linspace(100.0, 300.0, 80)
        result = macd(prices)
        valid = result.macd_line[~np.isnan(result.macd_line)]
        assert np.all(valid > 0)

    def test_custom_periods(self):
        prices = np.linspace(100.0, 150.0, 50)
        result = macd(prices, fast=5, slow=10, signal_period=3)
        valid = result.macd_line[~np.isnan(result.macd_line)]
        assert len(valid) > 0

    def test_downtrend_negative_macd(self):
        # Strong downtrend: fast EMA < slow EMA → MACD line should be negative
        prices = np.linspace(300.0, 100.0, 80)
        result = macd(prices)
        valid = result.macd_line[~np.isnan(result.macd_line)]
        assert np.all(valid < 0)

    def test_flat_series_macd_near_zero(self):
        # Flat prices → both EMAs converge to same value → MACD ≈ 0
        prices = np.full(80, 100.0)
        result = macd(prices)
        valid = result.macd_line[~np.isnan(result.macd_line)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_flat_series_histogram_near_zero(self):
        prices = np.full(80, 100.0)
        result = macd(prices)
        valid = result.histogram[~np.isnan(result.histogram)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_linear_ramp_constant_macd(self):
        # Linear ramp → both EMAs lag by a constant → MACD line should stabilise
        prices = np.linspace(100.0, 300.0, 200)
        result = macd(prices)
        valid = result.macd_line[~np.isnan(result.macd_line)]
        # After warmup, MACD should be positive and roughly constant
        tail = valid[-50:]
        assert np.all(tail > 0)
        assert np.std(tail) / np.mean(tail) < 0.05  # low relative variation

    def test_too_short_input(self):
        prices = np.array([100.0, 101.0, 102.0])
        result = macd(prices)
        assert np.all(np.isnan(result.macd_line))
        assert np.all(np.isnan(result.signal))
