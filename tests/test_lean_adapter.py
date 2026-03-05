"""Tests for the Lean adapter (LeanIndicatorAdapter and IndicatorLibrary)."""

import numpy as np
import pytest

from adapters.lean_adapter import LeanIndicatorAdapter, IndicatorLibrary


def _make_history(n: int = 100, seed: int = 42):
    """Create a dict mimicking a LEAN history DataFrame with OHLCV columns."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    spread = np.abs(rng.standard_normal(n)) * 2 + 0.5
    high = close + spread
    low = close - spread
    open_ = close + rng.standard_normal(n) * 0.5
    volume = np.abs(rng.standard_normal(n)) * 10000 + 5000

    class MockDF:
        """Minimal DataFrame-like object with column access and .columns."""
        def __init__(self, data):
            self._data = data
            self.columns = list(data.keys())

        def __getitem__(self, key):
            return self._data[key]

    return MockDF({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ------------------------------------------------------------------ #
#  LeanIndicatorAdapter                                                #
# ------------------------------------------------------------------ #

class TestLeanIndicatorAdapter:
    @pytest.fixture
    def adapter(self):
        return LeanIndicatorAdapter()

    @pytest.fixture
    def history(self):
        return _make_history()

    def test_trend_indicators(self, adapter, history):
        close = np.asarray(history["close"])
        assert len(adapter.efficiency_ratio(close)) == 100
        assert len(adapter.kama(close)) == 100
        assert len(adapter.sma(close)) == 100
        assert len(adapter.ema(close)) == 100
        assert len(adapter.wma(close)) == 100
        assert len(adapter.dema(close)) == 100
        assert len(adapter.tema(close)) == 100

    def test_linreg(self, adapter, history):
        close = np.asarray(history["close"])
        result = adapter.linreg(close)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"value", "slope", "intercept", "r_squared"}
        assert len(result["slope"]) == 100

    def test_linreg_forecast(self, adapter, history):
        close = np.asarray(history["close"])
        result = adapter.linreg_forecast(close)
        assert len(result) == 100

    def test_momentum_indicators(self, adapter, history):
        close = np.asarray(history["close"])
        high = np.asarray(history["high"])
        low = np.asarray(history["low"])
        assert len(adapter.roc(close)) == 100
        assert len(adapter.rsi(close)) == 100
        assert len(adapter.momentum(close)) == 100

    def test_macd(self, adapter, history):
        close = np.asarray(history["close"])
        result = adapter.macd(close)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"macd_line", "signal", "histogram"}

    def test_stochastic(self, adapter, history):
        result = adapter.stochastic(
            history["high"], history["low"], history["close"]
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"k", "d"}

    def test_volatility_indicators(self, adapter, history):
        high = np.asarray(history["high"])
        low = np.asarray(history["low"])
        close = np.asarray(history["close"])
        open_ = np.asarray(history["open"])
        assert len(adapter.true_range(high, low, close)) == 100
        assert len(adapter.atr(high, low, close)) == 100
        assert len(adapter.realized_vol(close)) == 100
        assert len(adapter.parkinson_vol(high, low)) == 100
        assert len(adapter.garman_klass_vol(open_, high, low, close)) == 100

    def test_range_indicators(self, adapter, history):
        high = np.asarray(history["high"])
        low = np.asarray(history["low"])
        close = np.asarray(history["close"])
        bb = adapter.bollinger_bands(close)
        assert isinstance(bb, dict)
        assert "bandwidth" in bb
        dc = adapter.donchian_channels(high, low)
        assert isinstance(dc, dict)
        assert "mid" in dc
        assert len(adapter.williams_r(high, low, close)) == 100
        assert len(adapter.price_zscore(close)) == 100

    def test_market_quality_indicators(self, adapter, history):
        close = np.asarray(history["close"])
        volume = np.asarray(history["volume"])
        assert len(adapter.fdi(close)) == 100
        assert len(adapter.hurst_exponent(close)) == 100
        assert len(adapter.price_entropy(close)) == 100
        assert len(adapter.volume_roc(volume)) == 100
        assert len(adapter.volume_zscore(volume)) == 100


# ------------------------------------------------------------------ #
#  IndicatorLibrary                                                    #
# ------------------------------------------------------------------ #

class TestIndicatorLibrary:
    @pytest.fixture
    def history(self):
        return _make_history(n=200)

    def test_compute_returns_dict(self, history):
        lib = IndicatorLibrary()
        result = lib.compute(history)
        assert isinstance(result, dict)

    def test_compute_contains_all_core_indicators(self, history):
        lib = IndicatorLibrary()
        result = lib.compute(history)

        expected_keys = [
            # Trend
            "efficiency_ratio", "kama", "sma", "ema",
            "linreg_slope", "linreg_intercept", "linreg_r_squared",
            # Momentum
            "roc", "rsi", "momentum",
            "macd", "macd_signal", "macd_histogram",
            "stochastic_k", "stochastic_d",
            # Volatility
            "atr", "true_range", "realized_vol", "parkinson_vol",
            "garman_klass_vol",
            # Range
            "bollinger_upper", "bollinger_middle", "bollinger_lower",
            "bollinger_bandwidth", "bollinger_percent_b",
            "donchian_upper", "donchian_lower", "donchian_mid",
            "williams_r", "price_zscore",
            # Market Quality
            "fdi", "hurst_exponent", "entropy",
            "volume_roc", "volume_zscore",
        ]
        for key in expected_keys:
            assert key in result, f"Missing indicator: {key}"

    def test_compute_arrays_correct_length(self, history):
        lib = IndicatorLibrary()
        result = lib.compute(history)
        for key, value in result.items():
            assert len(value) == 200, f"{key} has wrong length: {len(value)}"

    def test_compute_without_open(self):
        """When 'open' column is absent, garman_klass_vol should be skipped."""
        rng = np.random.default_rng(42)
        n = 200
        close = 100 + np.cumsum(rng.standard_normal(n))
        high = close + 2
        low = close - 2
        volume = np.abs(rng.standard_normal(n)) * 10000 + 5000

        class MockDF:
            def __init__(self, data):
                self._data = data
                self.columns = list(data.keys())
            def __getitem__(self, key):
                return self._data[key]

        history = MockDF({"high": high, "low": low, "close": close, "volume": volume})
        lib = IndicatorLibrary()
        result = lib.compute(history)
        assert "garman_klass_vol" not in result

    def test_compute_without_volume(self):
        """When 'volume' column is absent, volume indicators should be skipped."""
        rng = np.random.default_rng(42)
        n = 200
        close = 100 + np.cumsum(rng.standard_normal(n))
        high = close + 2
        low = close - 2
        open_ = close + rng.standard_normal(n) * 0.5

        class MockDF:
            def __init__(self, data):
                self._data = data
                self.columns = list(data.keys())
            def __getitem__(self, key):
                return self._data[key]

        history = MockDF({"open": open_, "high": high, "low": low, "close": close})
        lib = IndicatorLibrary()
        result = lib.compute(history)
        assert "volume_roc" not in result
        assert "volume_zscore" not in result

    def test_algorithm_stored(self):
        lib = IndicatorLibrary(algorithm="mock_algo")
        assert lib.algorithm == "mock_algo"

    def test_compute_rsi_matches_direct_call(self, history):
        """Verify IndicatorLibrary results match direct function calls."""
        lib = IndicatorLibrary()
        result = lib.compute(history)
        close = np.asarray(history["close"], dtype=float)
        import kaufman_indicators as ki
        expected = ki.rsi(close)
        np.testing.assert_array_equal(result["rsi"], expected)

    def test_compute_atr_matches_direct_call(self, history):
        lib = IndicatorLibrary()
        result = lib.compute(history)
        import kaufman_indicators as ki
        high = np.asarray(history["high"], dtype=float)
        low = np.asarray(history["low"], dtype=float)
        close = np.asarray(history["close"], dtype=float)
        expected = ki.atr(high, low, close)
        np.testing.assert_array_equal(result["atr"], expected)
