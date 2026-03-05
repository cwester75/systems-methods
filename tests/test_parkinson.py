"""Tests for Parkinson Volatility estimator."""

import numpy as np
import pytest

from kaufman_indicators.volatility.parkinson import parkinson_vol


class TestParkinsonVol:
    def test_returns_correct_length(self):
        n = 40
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        result = parkinson_vol(high, low, period=20)
        assert len(result) == n

    def test_first_period_minus_one_are_nan(self):
        n = 40
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        result = parkinson_vol(high, low, period=20)
        assert np.all(np.isnan(result[:19]))

    def test_no_nan_after_warmup(self):
        n = 40
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        result = parkinson_vol(high, low, period=20)
        assert not np.any(np.isnan(result[19:]))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.standard_normal(80))
        spread = np.abs(rng.standard_normal(80)) * 2 + 0.5
        high = base + spread
        low = base - spread
        result = parkinson_vol(high, low, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_constant_range_constant_vol(self):
        n = 40
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        result = parkinson_vol(high, low, period=10, annualize=False)
        valid = result[~np.isnan(result)]
        # All windows have the same H/L ratio → constant vol
        expected = np.sqrt(np.log(105.0 / 95.0) ** 2 / (4.0 * np.log(2.0)))
        np.testing.assert_allclose(valid, expected, atol=1e-10)

    def test_zero_range_zero_vol(self):
        # When high == low, ln(H/L) = 0 → vol = 0
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        result = parkinson_vol(high, low, period=10, annualize=False)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_annualization(self):
        n = 40
        high = np.full(n, 110.0)
        low = np.full(n, 90.0)
        ann = parkinson_vol(high, low, period=20, annualize=True)
        raw = parkinson_vol(high, low, period=20, annualize=False)
        ratio = ann[25] / raw[25]
        np.testing.assert_allclose(ratio, np.sqrt(252), atol=1e-10)

    def test_wider_range_higher_vol(self):
        n = 30
        high_narrow = np.full(n, 102.0)
        low_narrow = np.full(n, 98.0)
        high_wide = np.full(n, 110.0)
        low_wide = np.full(n, 90.0)
        vol_narrow = parkinson_vol(high_narrow, low_narrow, period=10, annualize=False)
        vol_wide = parkinson_vol(high_wide, low_wide, period=10, annualize=False)
        assert vol_wide[15] > vol_narrow[15]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            parkinson_vol(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_short_input(self):
        high = np.array([110.0, 112.0])
        low = np.array([90.0, 88.0])
        result = parkinson_vol(high, low, period=20)
        assert np.all(np.isnan(result))
