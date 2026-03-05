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

    def test_gap_down_increases_tr(self):
        # Close at 100, then gap-down: high=90, low=80 → |low-prev_close|=20
        high = np.array([105.0, 90.0])
        low = np.array([95.0, 80.0])
        close = np.array([100.0, 85.0])
        tr = true_range(high, low, close)
        # max(90-80=10, |90-100|=10, |80-100|=20) = 20
        assert tr[1] == pytest.approx(20.0)

    def test_known_values_multiple_bars(self):
        high = np.array([50.0, 48.0, 55.0])
        low = np.array([44.0, 42.0, 46.0])
        close = np.array([47.0, 43.0, 52.0])
        tr = true_range(high, low, close)
        # Bar 0: 50-44 = 6
        assert tr[0] == pytest.approx(6.0)
        # Bar 1: max(48-42=6, |48-47|=1, |42-47|=5) = 6
        assert tr[1] == pytest.approx(6.0)
        # Bar 2: max(55-46=9, |55-43|=12, |46-43|=3) = 12
        assert tr[2] == pytest.approx(12.0)

    def test_flat_prices_zero_tr(self):
        n = 10
        price = np.full(n, 100.0)
        tr = true_range(price, price, price)
        np.testing.assert_allclose(tr, 0.0, atol=1e-10)

    def test_tr_at_least_hl_diff(self):
        # True range is always >= high - low
        high, low, close = _make_ohlc(50)
        tr = true_range(high, low, close)
        hl = high - low
        assert np.all(tr >= hl - 1e-10)


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

    def test_flat_series_zero_atr(self):
        # When H == L == C (no range, no gaps), ATR should be 0
        n = 50
        price = np.full(n, 100.0)
        result = atr(price, price, price, period=14)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_noisy_series_higher_atr(self):
        # Noisier prices should produce higher ATR than calm prices
        n = 100
        calm_close = np.full(n, 100.0)
        calm_high = np.full(n, 101.0)
        calm_low = np.full(n, 99.0)
        rng = np.random.default_rng(42)
        noisy_close = 100 + np.cumsum(rng.standard_normal(n))
        noisy_spread = np.abs(rng.standard_normal(n)) * 5 + 1
        noisy_high = noisy_close + noisy_spread
        noisy_low = noisy_close - noisy_spread
        atr_calm = atr(calm_high, calm_low, calm_close, period=14)
        atr_noisy = atr(noisy_high, noisy_low, noisy_close, period=14)
        assert np.nanmean(atr_noisy) > np.nanmean(atr_calm)

    def test_trending_constant_spread_stable_atr(self):
        # Trending prices with constant H-L spread → ATR should be stable
        n = 80
        close = np.linspace(100, 200, n)
        high = close + 3.0
        low = close - 3.0
        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        # With constant spread of 6 and no gaps, ATR ≈ 6
        np.testing.assert_allclose(valid[-20:], 6.0, atol=0.5)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            atr(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))
