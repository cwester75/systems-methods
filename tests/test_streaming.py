"""Tests for streaming (stateful) indicator classes."""

import numpy as np
import pytest

import kaufman_indicators as ki
from kaufman_indicators.streaming import (
    Indicator,
    StreamingSMA,
    StreamingEMA,
    StreamingKAMA,
    StreamingROC,
    StreamingMomentum,
    StreamingRSI,
    StreamingMACD,
    StreamingATR,
    StreamingBollingerBands,
    STREAMING_INDICATORS,
    create_streaming,
)


# ── Test data ────────────────────────────────────────────────────────────────

@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    return 100 + np.cumsum(rng.standard_normal(200))


@pytest.fixture
def ohlc():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(200))
    spread = np.abs(rng.standard_normal(200)) * 2 + 0.5
    high = close + spread
    low = close - spread
    return high, low, close


# ── Base class ───────────────────────────────────────────────────────────────


class TestIndicatorBase:
    def test_not_ready_initially(self):
        ind = Indicator()
        assert not ind.is_ready
        assert ind.value is None
        assert ind.count == 0

    def test_update_raises(self):
        ind = Indicator()
        with pytest.raises(NotImplementedError):
            ind.update(1.0)

    def test_repr(self):
        ind = Indicator(name="Test")
        assert "warming" in repr(ind)
        assert "Test" in repr(ind)

    def test_reset(self):
        ind = Indicator()
        ind._value = 42.0
        ind._count = 10
        ind.reset()
        assert ind.value is None
        assert ind.count == 0


# ── SMA ──────────────────────────────────────────────────────────────────────


class TestStreamingSMA:
    def test_warmup_period(self):
        sma = StreamingSMA(period=5)
        for i in range(4):
            assert sma.update(float(i)) is None
        assert not sma.is_ready

    def test_first_value(self):
        sma = StreamingSMA(period=3)
        sma.update(1.0)
        sma.update(2.0)
        val = sma.update(3.0)
        assert val == pytest.approx(2.0)

    def test_matches_batch(self, prices):
        period = 20
        sma = StreamingSMA(period=period)
        batch = ki.sma(prices, period)

        for i, p in enumerate(prices):
            val = sma.update(p)
            if i >= period - 1:
                assert val == pytest.approx(batch[i], rel=1e-10)

    def test_reset(self):
        sma = StreamingSMA(period=3)
        for p in [1.0, 2.0, 3.0]:
            sma.update(p)
        assert sma.is_ready
        sma.reset()
        assert not sma.is_ready
        assert sma.count == 0


# ── EMA ──────────────────────────────────────────────────────────────────────


class TestStreamingEMA:
    def test_warmup_period(self):
        ema = StreamingEMA(period=10)
        for i in range(9):
            assert ema.update(float(i)) is None

    def test_first_value_is_sma(self):
        ema = StreamingEMA(period=3)
        ema.update(1.0)
        ema.update(2.0)
        val = ema.update(3.0)
        assert val == pytest.approx(2.0)

    def test_matches_batch(self, prices):
        period = 20
        ema = StreamingEMA(period=period)
        batch = ki.ema(prices, period)

        for i, p in enumerate(prices):
            val = ema.update(p)
            if i >= period - 1:
                assert val == pytest.approx(batch[i], rel=1e-10)

    def test_custom_alpha(self):
        ema = StreamingEMA(period=3, alpha=0.5)
        ema.update(10.0)
        ema.update(20.0)
        seed = ema.update(30.0)  # SMA seed = 20
        assert seed == pytest.approx(20.0)
        val = ema.update(40.0)
        assert val == pytest.approx(20.0 + 0.5 * (40.0 - 20.0))


# ── KAMA ─────────────────────────────────────────────────────────────────────


class TestStreamingKAMA:
    def test_warmup_period(self):
        kama = StreamingKAMA(period=10)
        for i in range(10):
            assert kama.update(float(i)) is None

    def test_matches_batch(self, prices):
        period = 10
        kama_s = StreamingKAMA(period=period)
        batch = ki.kama(prices, period)

        for i, p in enumerate(prices):
            val = kama_s.update(p)
            if i >= period and not np.isnan(batch[i]):
                assert val == pytest.approx(batch[i], rel=1e-8), f"Mismatch at {i}"


# ── ROC ──────────────────────────────────────────────────────────────────────


class TestStreamingROC:
    def test_warmup(self):
        roc = StreamingROC(period=5)
        for i in range(5):
            assert roc.update(float(100 + i)) is None

    def test_basic_value(self):
        roc = StreamingROC(period=3)
        roc.update(100.0)
        roc.update(101.0)
        roc.update(102.0)
        val = roc.update(110.0)
        assert val == pytest.approx((110.0 / 100.0 - 1.0) * 100.0)

    def test_matches_batch(self, prices):
        period = 12
        roc = StreamingROC(period=period)
        batch = ki.roc(prices, period)

        for i, p in enumerate(prices):
            val = roc.update(p)
            if i >= period and not np.isnan(batch[i]):
                assert val == pytest.approx(batch[i], rel=1e-10)


# ── Momentum ─────────────────────────────────────────────────────────────────


class TestStreamingMomentum:
    def test_basic_value(self):
        mom = StreamingMomentum(period=3)
        mom.update(100.0)
        mom.update(101.0)
        mom.update(102.0)
        val = mom.update(110.0)
        assert val == pytest.approx(10.0)

    def test_matches_batch(self, prices):
        period = 10
        mom = StreamingMomentum(period=period)
        batch = ki.momentum(prices, period)

        for i, p in enumerate(prices):
            val = mom.update(p)
            if i >= period and not np.isnan(batch[i]):
                assert val == pytest.approx(batch[i], rel=1e-10)


# ── RSI ──────────────────────────────────────────────────────────────────────


class TestStreamingRSI:
    def test_warmup(self):
        rsi = StreamingRSI(period=14)
        for i in range(14):
            assert rsi.update(float(100 + i)) is None

    def test_matches_batch(self, prices):
        period = 14
        rsi = StreamingRSI(period=period)
        batch = ki.rsi(prices, period)

        for i, p in enumerate(prices):
            val = rsi.update(p)
            if val is not None and not np.isnan(batch[i]):
                assert val == pytest.approx(batch[i], rel=1e-10), (
                    f"Mismatch at index {i}: streaming={val}, batch={batch[i]}"
                )

    def test_range_0_100(self, prices):
        rsi = StreamingRSI(period=14)
        for p in prices:
            val = rsi.update(p)
            if val is not None:
                assert 0.0 <= val <= 100.0

    def test_all_gains_is_100(self):
        rsi = StreamingRSI(period=5)
        for i in range(10):
            rsi.update(float(100 + i))
        assert rsi.value == pytest.approx(100.0)


# ── MACD ─────────────────────────────────────────────────────────────────────


class TestStreamingMACD:
    def test_warmup(self):
        macd = StreamingMACD(fast=12, slow=26, signal_period=9)
        for i in range(25):
            assert macd.update(float(100 + i)) is None

    def test_components_available(self, prices):
        macd = StreamingMACD()
        for p in prices:
            macd.update(p)
        assert macd.value is not None
        assert macd.signal_value is not None
        assert macd.histogram_value is not None
        assert macd.histogram_value == pytest.approx(macd.value - macd.signal_value)

    def test_reset_clears_all(self):
        macd = StreamingMACD()
        for i in range(50):
            macd.update(float(100 + i * 0.5))
        macd.reset()
        assert macd.value is None
        assert macd.signal_value is None
        assert macd.histogram_value is None


# ── ATR ──────────────────────────────────────────────────────────────────────


class TestStreamingATR:
    def test_warmup(self):
        atr = StreamingATR(period=14)
        for i in range(13):
            assert atr.update(100.0, 102.0, 98.0) is None

    def test_matches_batch(self, ohlc):
        high, low, close = ohlc
        period = 14
        atr_s = StreamingATR(period=period)
        batch = ki.atr(high, low, close, period)

        for i in range(len(close)):
            val = atr_s.update(close[i], high[i], low[i])
            if i >= period - 1 and not np.isnan(batch[i]):
                assert val == pytest.approx(batch[i], rel=1e-10), (
                    f"Mismatch at index {i}"
                )

    def test_close_only_fallback(self):
        atr = StreamingATR(period=3)
        atr.update(100.0)  # TR = 0 (high-low=0, no prev close)
        atr.update(101.0)  # TR = |101-100| = 1 (gap from prev close)
        val = atr.update(102.0)  # TR = |102-101| = 1
        assert val is not None
        # Mean of [0, 1, 1] = 0.6667
        assert val == pytest.approx(2.0 / 3.0)


# ── Bollinger Bands ──────────────────────────────────────────────────────────


class TestStreamingBollingerBands:
    def test_warmup(self):
        bb = StreamingBollingerBands(period=20)
        for i in range(19):
            assert bb.update(float(100 + i)) is None

    def test_components(self, prices):
        bb = StreamingBollingerBands(period=20, num_std=2.0)
        for p in prices:
            bb.update(p)
        assert bb.value is not None
        assert bb.upper is not None
        assert bb.lower is not None
        assert bb.bandwidth is not None
        assert bb.percent_b is not None
        assert bb.upper > bb.value > bb.lower

    def test_reset(self):
        bb = StreamingBollingerBands(period=5)
        for i in range(10):
            bb.update(float(100 + i))
        bb.reset()
        assert bb.value is None
        assert bb.upper is None
        assert bb.lower is None


# ── Factory ──────────────────────────────────────────────────────────────────


class TestFactory:
    def test_create_streaming(self):
        rsi = create_streaming("rsi", period=14)
        assert isinstance(rsi, StreamingRSI)

    def test_create_streaming_unknown(self):
        with pytest.raises(KeyError, match="Unknown streaming indicator"):
            create_streaming("nonexistent")

    def test_all_registry_entries_are_indicator_subclasses(self):
        for name, cls in STREAMING_INDICATORS.items():
            assert issubclass(cls, Indicator), f"{name} is not an Indicator subclass"

    def test_create_all(self):
        for name in STREAMING_INDICATORS:
            # All should be constructable with defaults (except sma/ema/wma need period)
            if name in ("sma", "ema"):
                ind = create_streaming(name, period=10)
            else:
                ind = create_streaming(name)
            assert isinstance(ind, Indicator)
            assert not ind.is_ready


# ── Import from top-level ────────────────────────────────────────────────────


class TestTopLevelImports:
    def test_indicator_base_importable(self):
        from kaufman_indicators import Indicator
        assert Indicator is not None

    def test_streaming_sma_importable(self):
        from kaufman_indicators import StreamingSMA
        assert StreamingSMA is not None

    def test_create_streaming_importable(self):
        from kaufman_indicators import create_streaming
        assert create_streaming is not None

    def test_streaming_indicators_dict_importable(self):
        from kaufman_indicators import STREAMING_INDICATORS
        assert len(STREAMING_INDICATORS) > 0
