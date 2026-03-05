"""Tests for Volume Rate of Change and Volume Z-Score."""

import numpy as np
import pytest

from kaufman_indicators.market_quality.volume_roc import volume_roc
from kaufman_indicators.market_quality.volume_zscore import volume_zscore


class TestVolumeROC:
    def test_returns_correct_length(self):
        volume = np.arange(1.0, 31.0) * 1000
        result = volume_roc(volume, period=12)
        assert len(result) == 30

    def test_first_period_values_are_nan(self):
        volume = np.arange(1.0, 31.0) * 1000
        result = volume_roc(volume, period=12)
        assert np.all(np.isnan(result[:12]))

    def test_no_nan_after_warmup(self):
        volume = np.arange(1.0, 31.0) * 1000
        result = volume_roc(volume, period=12)
        assert not np.any(np.isnan(result[12:]))

    def test_known_value(self):
        # vol[0]=1000, vol[5]=2000 → ROC = (2000-1000)/1000 * 100 = 100%
        volume = np.array([1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0])
        result = volume_roc(volume, period=5)
        np.testing.assert_allclose(result[5], 100.0, atol=1e-10)

    def test_constant_volume_zero_roc(self):
        volume = np.full(30, 50000.0)
        result = volume_roc(volume, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_increasing_volume_positive(self):
        volume = np.linspace(1000, 5000, 30)
        result = volume_roc(volume, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_decreasing_volume_negative(self):
        volume = np.linspace(5000, 1000, 30)
        result = volume_roc(volume, period=10)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)

    def test_zero_volume_handled(self):
        volume = np.array([0.0, 0.0, 0.0, 100.0, 200.0, 300.0])
        result = volume_roc(volume, period=3)
        assert np.isnan(result[3])  # 100 vs 0 → NaN

    def test_too_short_input(self):
        volume = np.array([1000.0, 2000.0])
        result = volume_roc(volume, period=5)
        assert np.all(np.isnan(result))

    def test_default_period(self):
        volume = np.arange(1.0, 31.0) * 1000
        result = volume_roc(volume)  # default period=12
        assert np.all(np.isnan(result[:12]))
        assert not np.any(np.isnan(result[12:]))

    def test_doubling_volume_100_percent(self):
        # If volume doubles every period, ROC should be 100%
        volume = np.array([1000.0, 1100.0, 1200.0, 1300.0, 2000.0])
        result = volume_roc(volume, period=4)
        # vol[4]=2000, vol[0]=1000 → (2000-1000)/1000*100 = 100
        np.testing.assert_allclose(result[4], 100.0, atol=1e-10)

    def test_halving_volume_minus_50(self):
        volume = np.array([2000.0, 1800.0, 1600.0, 1000.0])
        result = volume_roc(volume, period=3)
        # vol[3]=1000, vol[0]=2000 → (1000-2000)/2000*100 = -50
        np.testing.assert_allclose(result[3], -50.0, atol=1e-10)


class TestVolumeZScore:
    def test_returns_correct_length(self):
        volume = np.arange(1.0, 41.0) * 1000
        result = volume_zscore(volume, period=20)
        assert len(result) == 40

    def test_first_period_minus_one_are_nan(self):
        volume = np.arange(1.0, 41.0) * 1000
        result = volume_zscore(volume, period=20)
        assert np.all(np.isnan(result[:19]))

    def test_no_nan_after_warmup(self):
        volume = np.arange(1.0, 41.0) * 1000
        result = volume_zscore(volume, period=20)
        assert not np.any(np.isnan(result[19:]))

    def test_constant_volume_zero_zscore(self):
        volume = np.full(30, 50000.0)
        result = volume_zscore(volume, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_value(self):
        volume = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        result = volume_zscore(volume, period=5)
        mean = np.mean(volume)
        std = np.std(volume, ddof=1)
        expected = (volume[-1] - mean) / std
        np.testing.assert_allclose(result[4], expected, atol=1e-10)

    def test_spike_detection(self):
        # A sudden volume spike should produce a high z-score
        volume = np.full(30, 1000.0)
        volume[29] = 5000.0  # spike
        result = volume_zscore(volume, period=20)
        assert result[29] > 2.0  # should be very high

    def test_positive_when_above_mean(self):
        # Increasing volume → recent values above mean → positive z-score
        volume = np.linspace(1000, 5000, 40)
        result = volume_zscore(volume, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_negative_when_below_mean(self):
        # Decreasing volume → recent values below mean → negative z-score
        volume = np.linspace(5000, 1000, 40)
        result = volume_zscore(volume, period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid < 0)

    def test_too_short_input(self):
        volume = np.array([1000.0, 2000.0])
        result = volume_zscore(volume, period=20)
        assert np.all(np.isnan(result))

    def test_default_period(self):
        volume = np.arange(1.0, 41.0) * 1000
        result = volume_zscore(volume)  # default period=20
        assert np.all(np.isnan(result[:19]))
        assert not np.any(np.isnan(result[19:]))

    def test_low_volume_negative_zscore(self):
        # Constant volume then sudden drop → negative z-score
        volume = np.full(30, 5000.0)
        volume[29] = 1000.0  # drop
        result = volume_zscore(volume, period=20)
        assert result[29] < -2.0

    def test_symmetry(self):
        # Z-scores from symmetric data around the mean should be near zero
        volume = np.array([90.0, 110.0] * 15)  # mean=100, alternating
        result = volume_zscore(volume, period=20)
        # Last value 110 should be positive, second to last 90 should be negative
        assert result[-1] > 0
        assert result[-2] < 0
