"""Tests for pd.Series output standardization.

When indicators receive a pd.Series input, the output should be a pd.Series
with the original index preserved and the name attribute set.

When indicators receive a plain numpy array, output remains a numpy array
(backward compatible).
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import kaufman_indicators as ki


# ── Helper: date-indexed Series ──────────────────────────────────────────────

def _make_series(n: int = 50, name: str = "close", seed: int = 42) -> pd.Series:
    """Return a pd.Series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    data = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.Series(data, index=index, name=name)


def _make_ohlcv(n: int = 50, seed: int = 42):
    """Return high, low, close, open, volume as pd.Series with shared index."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    spread = np.abs(rng.standard_normal(n)) * 0.3 + 0.2
    high = pd.Series(close + spread, index=index, name="high")
    low = pd.Series(close - spread, index=index, name="low")
    close_s = pd.Series(close, index=index, name="close")
    open_s = pd.Series(close + rng.standard_normal(n) * 0.1, index=index, name="open")
    volume = pd.Series(np.abs(rng.standard_normal(n)) * 1e6 + 1e5, index=index, name="volume")
    return high, low, close_s, open_s, volume


# ── Single-output indicators: pd.Series in → pd.Series out ──────────────────

class TestSingleOutputSeries:
    """Indicators that return a single array should return pd.Series when
    given pd.Series input."""

    def test_sma_returns_series(self):
        prices = _make_series()
        result = ki.sma(prices, 20)
        assert isinstance(result, pd.Series)
        assert result.name == "sma"
        assert result.index.equals(prices.index)

    def test_ema_returns_series(self):
        prices = _make_series()
        result = ki.ema(prices, 20)
        assert isinstance(result, pd.Series)
        assert result.name == "ema"
        assert result.index.equals(prices.index)

    def test_rsi_returns_series(self):
        prices = _make_series()
        result = ki.rsi(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "rsi"
        assert result.index.equals(prices.index)

    def test_roc_returns_series(self):
        prices = _make_series()
        result = ki.roc(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "roc"

    def test_momentum_returns_series(self):
        prices = _make_series()
        result = ki.momentum(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "momentum"

    def test_efficiency_ratio_returns_series(self):
        prices = _make_series()
        result = ki.efficiency_ratio(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "efficiency_ratio"

    def test_kama_returns_series(self):
        prices = _make_series()
        result = ki.kama(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "kama"

    def test_atr_returns_series(self):
        high, low, close, _, _ = _make_ohlcv()
        result = ki.atr(high, low, close)
        assert isinstance(result, pd.Series)
        assert result.name == "atr"
        assert result.index.equals(high.index)

    def test_true_range_returns_series(self):
        high, low, close, _, _ = _make_ohlcv()
        result = ki.true_range(high, low, close)
        assert isinstance(result, pd.Series)
        assert result.name == "true_range"

    def test_williams_r_returns_series(self):
        high, low, close, _, _ = _make_ohlcv()
        result = ki.williams_r(high, low, close)
        assert isinstance(result, pd.Series)
        assert result.name == "williams_r"

    def test_price_zscore_returns_series(self):
        prices = _make_series()
        result = ki.price_zscore(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "price_zscore"

    def test_volume_roc_returns_series(self):
        _, _, _, _, volume = _make_ohlcv()
        result = ki.volume_roc(volume)
        assert isinstance(result, pd.Series)
        assert result.name == "volume_roc"

    def test_volume_zscore_returns_series(self):
        _, _, _, _, volume = _make_ohlcv()
        result = ki.volume_zscore(volume)
        assert isinstance(result, pd.Series)
        assert result.name == "volume_zscore"

    def test_realized_vol_returns_series(self):
        prices = _make_series()
        result = ki.realized_vol(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "realized_vol"

    def test_parkinson_vol_returns_series(self):
        high, low, _, _, _ = _make_ohlcv()
        result = ki.parkinson_vol(high, low)
        assert isinstance(result, pd.Series)
        assert result.name == "parkinson_vol"

    def test_garman_klass_vol_returns_series(self):
        high, low, close, open_s, _ = _make_ohlcv()
        result = ki.garman_klass_vol(open_s, high, low, close)
        assert isinstance(result, pd.Series)
        assert result.name == "garman_klass_vol"

    def test_fdi_returns_series(self):
        prices = _make_series()
        result = ki.fdi(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "fdi"

    def test_hurst_exponent_returns_series(self):
        prices = _make_series(n=150)
        result = ki.hurst_exponent(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "hurst_exponent"

    def test_price_entropy_returns_series(self):
        prices = _make_series(n=100)
        result = ki.price_entropy(prices)
        assert isinstance(result, pd.Series)
        assert result.name == "price_entropy"


# ── Multi-output indicators: NamedTuple fields become pd.Series ─────────────

class TestMultiOutputSeries:
    """NamedTuple-returning indicators should wrap each field as pd.Series."""

    def test_macd_returns_series_fields(self):
        prices = _make_series()
        result = ki.macd(prices)
        assert isinstance(result.macd_line, pd.Series)
        assert result.macd_line.name == "macd_macd_line"
        assert isinstance(result.signal, pd.Series)
        assert isinstance(result.histogram, pd.Series)
        assert result.macd_line.index.equals(prices.index)

    def test_bollinger_returns_series_fields(self):
        prices = _make_series()
        result = ki.bollinger_bands(prices)
        assert isinstance(result.middle, pd.Series)
        assert result.middle.name == "bollinger_bands_middle"
        assert isinstance(result.upper, pd.Series)
        assert isinstance(result.lower, pd.Series)
        assert result.upper.index.equals(prices.index)

    def test_stochastic_returns_series_fields(self):
        high, low, close, _, _ = _make_ohlcv()
        result = ki.stochastic(high, low, close)
        assert isinstance(result.k, pd.Series)
        assert result.k.name == "stochastic_k"
        assert isinstance(result.d, pd.Series)
        assert result.k.index.equals(high.index)

    def test_donchian_returns_series_fields(self):
        high, low, _, _, _ = _make_ohlcv()
        result = ki.donchian_channels(high, low)
        assert isinstance(result.upper, pd.Series)
        assert result.upper.name == "donchian_channels_upper"
        assert result.upper.index.equals(high.index)

    def test_linreg_returns_series_fields(self):
        prices = _make_series()
        result = ki.linreg(prices)
        assert isinstance(result.value, pd.Series)
        assert result.value.name == "linreg_value"
        assert isinstance(result.slope, pd.Series)
        assert result.slope.index.equals(prices.index)


# ── Numpy backward compatibility ────────────────────────────────────────────

class TestNumpyBackwardCompat:
    """When plain numpy arrays are passed, output must remain numpy arrays."""

    def test_sma_numpy_in_numpy_out(self):
        prices = np.linspace(100, 150, 40)
        result = ki.sma(prices, 20)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, pd.Series)

    def test_rsi_numpy_in_numpy_out(self):
        prices = np.linspace(100, 150, 40)
        result = ki.rsi(prices)
        assert isinstance(result, np.ndarray)

    def test_macd_numpy_in_numpy_out(self):
        prices = np.linspace(100, 150, 50)
        result = ki.macd(prices)
        assert isinstance(result.macd_line, np.ndarray)
        assert not isinstance(result.macd_line, pd.Series)

    def test_true_range_numpy_in_numpy_out(self):
        high = np.array([110.0, 112.0])
        low = np.array([105.0, 108.0])
        close = np.array([108.0, 110.0])
        result = ki.true_range(high, low, close)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, pd.Series)

    def test_list_input_numpy_out(self):
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        result = ki.momentum(prices, period=2)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, pd.Series)


# ── Values unchanged ────────────────────────────────────────────────────────

class TestValuesPreserved:
    """Numeric results must be identical whether input is Series or array."""

    def test_rsi_values_match(self):
        arr = np.linspace(100, 150, 40)
        series = pd.Series(arr)
        result_np = ki.rsi(arr)
        result_pd = ki.rsi(series)
        np.testing.assert_allclose(result_pd.values, result_np, atol=1e-14)

    def test_bollinger_values_match(self):
        arr = np.linspace(100, 150, 40)
        series = pd.Series(arr)
        bb_np = ki.bollinger_bands(arr)
        bb_pd = ki.bollinger_bands(series)
        np.testing.assert_allclose(bb_pd.middle.values, bb_np.middle, atol=1e-14)
        np.testing.assert_allclose(bb_pd.upper.values, bb_np.upper, atol=1e-14)

    def test_momentum_values_match(self):
        arr = np.arange(1.0, 31.0)
        series = pd.Series(arr)
        result_np = ki.momentum(arr, period=10)
        result_pd = ki.momentum(series, period=10)
        np.testing.assert_allclose(result_pd.values, result_np, atol=1e-14)


# ── Index preservation with custom index ────────────────────────────────────

class TestCustomIndex:
    """The output index should match the input, not be a default RangeIndex."""

    def test_string_index_preserved(self):
        idx = [f"bar_{i}" for i in range(30)]
        prices = pd.Series(np.linspace(100, 150, 30), index=idx)
        result = ki.sma(prices, 10)
        assert list(result.index) == idx

    def test_datetime_index_preserved(self):
        prices = _make_series(n=30)
        result = ki.efficiency_ratio(prices, period=10)
        assert result.index.equals(prices.index)

    def test_integer_index_preserved(self):
        idx = list(range(100, 130))
        prices = pd.Series(np.linspace(100, 150, 30), index=idx)
        result = ki.roc(prices, period=5)
        assert list(result.index) == idx
