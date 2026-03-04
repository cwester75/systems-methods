"""Tests for the Average True Range (ATR) and True Range (TR)."""

import numpy as np
import pytest

from kaufman_indicators.volatility.true_range import true_range
from kaufman_indicators.volatility.atr import atr


def _make_ohlc(n: int, seed: int = 0):
    """Return synthetic high, low, close arrays of length *n*."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    spread = np.abs(rng.standard_normal(n)) + 0.5
    high = close + spread
    low = close - spread
    return high, low, close


class TestTrueRange:
    def test_returns_correct_length(self):
        high, low, close = _make_ohlc(30)
        tr = true_range(high, low, close)
        assert len(tr) == 30

    def test_first_bar_is_hl_diff(self):
        high = np.array([110.0, 112.0])
        low = np.array([105.0, 108.0])
        close = np.array([108.0, 110.0])
        tr = true_range(high, low, close)
        assert tr[0] == pytest.approx(5.0)

    def test_tr_non_negative(self):
        high, low, close = _make_ohlc(50)
        tr = true_range(high, low, close)
        assert np.all(tr >= 0)

    def test_gap_up_increases_tr(self):
        # Close at 100, then gap-up: high=120, low=110 → TR should be 20
        high = np.array([105.0, 120.0])
        low = np.array([95.0, 110.0])
        close = np.array([100.0, 115.0])
        tr = true_range(high, low, close)
        # max(high[1]-low[1]=10, |high[1]-close[0]|=20, |low[1]-close[0]|=10) = 20
        assert tr[1] == pytest.approx(20.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            true_range(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))


class TestATR:
    def test_returns_correct_length(self):
        high, low, close = _make_ohlc(50)
        result = atr(high, low, close, period=14)
        assert len(result) == 50

    def test_first_period_minus_one_are_nan(self):
        high, low, close = _make_ohlc(50)
        result = atr(high, low, close, period=14)
        assert np.all(np.isnan(result[:13]))

    def test_atr_non_negative(self):
        high, low, close = _make_ohlc(100)
        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0)

    def test_constant_range_gives_constant_atr(self):
        n = 50
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 10.0, atol=1e-6)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            atr(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))
