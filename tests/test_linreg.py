"""Tests for Linear Regression indicators (slope, intercept, R², value, forecast)."""

import numpy as np
import pytest

from kaufman_indicators.trend.linreg import linreg, linreg_forecast


class TestLinReg:
    def test_returns_correct_length(self):
        prices = np.arange(1.0, 31.0)
        result = linreg(prices, period=14)
        assert len(result.value) == 30
        assert len(result.slope) == 30
        assert len(result.intercept) == 30
        assert len(result.r_squared) == 30

    def test_first_period_minus_one_are_nan(self):
        prices = np.arange(1.0, 31.0)
        result = linreg(prices, period=14)
        assert np.all(np.isnan(result.value[:13]))
        assert np.all(np.isnan(result.slope[:13]))
        assert np.all(np.isnan(result.intercept[:13]))
        assert np.all(np.isnan(result.r_squared[:13]))

    def test_no_nan_after_warmup(self):
        prices = np.arange(1.0, 31.0)
        result = linreg(prices, period=14)
        assert not np.any(np.isnan(result.value[13:]))
        assert not np.any(np.isnan(result.slope[13:]))
        assert not np.any(np.isnan(result.intercept[13:]))
        assert not np.any(np.isnan(result.r_squared[13:]))

    def test_perfect_linear_trend_slope(self):
        # y = 2*x + 5 → slope should be 2.0
        x = np.arange(50, dtype=float)
        prices = 2.0 * x + 5.0
        result = linreg(prices, period=14)
        valid_slopes = result.slope[~np.isnan(result.slope)]
        np.testing.assert_allclose(valid_slopes, 2.0, atol=1e-10)

    def test_perfect_linear_trend_intercept(self):
        # y = 2*x + 5 over a rolling window of 14 bars
        # For window ending at index i, x_local = [0..13], y = 2*(i-13+x_local)+5
        # intercept_local = y_mean - slope * x_mean
        x = np.arange(50, dtype=float)
        prices = 2.0 * x + 5.0
        result = linreg(prices, period=14)
        # For window ending at bar i: y_local[j] = 2*(i-13+j) + 5
        # slope = 2, intercept = 2*(i-13) + 5
        for i in range(13, 50):
            expected_intercept = 2.0 * (i - 13) + 5.0
            np.testing.assert_allclose(result.intercept[i], expected_intercept, atol=1e-10)

    def test_perfect_linear_trend_r_squared(self):
        # Perfect linear data → R² = 1.0
        prices = np.linspace(10.0, 100.0, 40)
        result = linreg(prices, period=14)
        valid = result.r_squared[~np.isnan(result.r_squared)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_constant_prices_r_squared(self):
        # Constant prices → R² = 1.0 (no variance to explain)
        prices = np.full(30, 50.0)
        result = linreg(prices, period=10)
        valid = result.r_squared[~np.isnan(result.r_squared)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_constant_prices_slope_zero(self):
        prices = np.full(30, 42.0)
        result = linreg(prices, period=10)
        valid = result.slope[~np.isnan(result.slope)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_r_squared_bounded_zero_to_one(self):
        rng = np.random.default_rng(99)
        prices = 100.0 + np.cumsum(rng.standard_normal(200))
        result = linreg(prices, period=20)
        valid = result.r_squared[~np.isnan(result.r_squared)]
        assert np.all(valid >= -1e-10)
        assert np.all(valid <= 1.0 + 1e-10)

    def test_value_equals_slope_times_last_x_plus_intercept(self):
        rng = np.random.default_rng(7)
        prices = 50 + np.cumsum(rng.standard_normal(40))
        period = 10
        result = linreg(prices, period)
        for i in range(period - 1, len(prices)):
            expected = result.slope[i] * (period - 1) + result.intercept[i]
            np.testing.assert_allclose(result.value[i], expected, atol=1e-10)

    def test_too_short_input(self):
        prices = np.array([1.0, 2.0, 3.0])
        result = linreg(prices, period=14)
        assert np.all(np.isnan(result.value))
        assert np.all(np.isnan(result.slope))
        assert np.all(np.isnan(result.intercept))
        assert np.all(np.isnan(result.r_squared))


class TestLinRegForecast:
    def test_forecast_offset_one(self):
        prices = np.linspace(10.0, 50.0, 30)
        result = linreg(prices, period=14)
        forecast = linreg_forecast(prices, period=14, offset=1)
        valid_idx = ~np.isnan(result.value)
        np.testing.assert_allclose(
            forecast[valid_idx],
            result.value[valid_idx] + result.slope[valid_idx],
            atol=1e-10,
        )

    def test_forecast_offset_zero_equals_value(self):
        prices = np.linspace(10.0, 50.0, 30)
        result = linreg(prices, period=14)
        forecast = linreg_forecast(prices, period=14, offset=0)
        valid_idx = ~np.isnan(result.value)
        np.testing.assert_allclose(
            forecast[valid_idx], result.value[valid_idx], atol=1e-10
        )
