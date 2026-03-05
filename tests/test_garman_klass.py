"""Tests for Garman-Klass Volatility estimator."""

import numpy as np
import pytest

from kaufman_indicators.volatility.garman_klass import garman_klass_vol


class TestGarmanKlassVol:
    def test_returns_correct_length(self):
        n = 40
        o = np.full(n, 100.0)
        h = np.full(n, 110.0)
        l = np.full(n, 90.0)
        c = np.full(n, 100.0)
        result = garman_klass_vol(o, h, l, c, period=20)
        assert len(result) == n

    def test_first_period_minus_one_are_nan(self):
        n = 40
        o = np.full(n, 100.0)
        h = np.full(n, 110.0)
        l = np.full(n, 90.0)
        c = np.full(n, 100.0)
        result = garman_klass_vol(o, h, l, c, period=20)
        assert np.all(np.isnan(result[:19]))

    def test_no_nan_after_warmup(self):
        n = 40
        o = np.full(n, 100.0)
        h = np.full(n, 110.0)
        l = np.full(n, 90.0)
        c = np.full(n, 100.0)
        result = garman_klass_vol(o, h, l, c, period=20)
        assert not np.any(np.isnan(result[19:]))

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        base = 100 + np.cumsum(rng.standard_normal(80) * 0.5)
        spread = np.abs(rng.standard_normal(80)) * 2 + 0.5
        o = base
        h = base + spread
        l = base - spread
        c = base + rng.standard_normal(80) * 0.3
        result = garman_klass_vol(o, h, l, c, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_close_equals_open_reduces_to_hl_component(self):
        # When C == O: ln(C/O) = 0, so GK = sqrt(0.5 * mean(ln(H/L)²))
        n = 30
        o = np.full(n, 100.0)
        h = np.full(n, 110.0)
        l = np.full(n, 90.0)
        c = np.full(n, 100.0)  # C == O
        result = garman_klass_vol(o, h, l, c, period=10, annualize=False)
        log_hl = np.log(110.0 / 90.0)
        expected = np.sqrt(0.5 * log_hl ** 2)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, expected, atol=1e-10)

    def test_annualization(self):
        n = 40
        o = np.full(n, 100.0)
        h = np.full(n, 110.0)
        l = np.full(n, 90.0)
        c = np.full(n, 105.0)
        ann = garman_klass_vol(o, h, l, c, period=20, annualize=True)
        raw = garman_klass_vol(o, h, l, c, period=20, annualize=False)
        ratio = ann[25] / raw[25]
        np.testing.assert_allclose(ratio, np.sqrt(252), atol=1e-10)

    def test_wider_range_higher_vol(self):
        n = 30
        o = np.full(n, 100.0)
        c = np.full(n, 100.0)
        h_narrow = np.full(n, 102.0)
        l_narrow = np.full(n, 98.0)
        h_wide = np.full(n, 115.0)
        l_wide = np.full(n, 85.0)
        vol_narrow = garman_klass_vol(o, h_narrow, l_narrow, c, period=10, annualize=False)
        vol_wide = garman_klass_vol(o, h_wide, l_wide, c, period=10, annualize=False)
        assert vol_wide[15] > vol_narrow[15]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            garman_klass_vol(
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0]),
            )

    def test_too_short_input(self):
        o = np.array([100.0, 100.0])
        h = np.array([110.0, 112.0])
        l = np.array([90.0, 88.0])
        c = np.array([105.0, 106.0])
        result = garman_klass_vol(o, h, l, c, period=20)
        assert np.all(np.isnan(result))
