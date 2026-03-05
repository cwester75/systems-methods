"""Tests for the indicator metadata registry and parameter schema."""

import numpy as np
import pytest

from kaufman_indicators.registry import INDICATORS
from kaufman_indicators.registry_meta import (
    INDICATOR_META,
    get_meta,
    list_by_category,
    list_by_input,
    schema,
    defaults,
    required_params,
    output_fields,
    validate_meta,
)


class TestIndicatorMeta:
    def test_all_indicators_covered(self):
        assert set(INDICATOR_META.keys()) == set(INDICATORS.keys())

    def test_every_entry_has_required_keys(self):
        for name, meta in INDICATOR_META.items():
            assert "category" in meta, f"{name} missing category"
            assert "inputs" in meta, f"{name} missing inputs"
            assert "params" in meta, f"{name} missing params"
            assert "output" in meta, f"{name} missing output"

    def test_categories_are_valid(self):
        valid = {"trend", "momentum", "volatility", "range", "market_quality"}
        for name, meta in INDICATOR_META.items():
            assert meta["category"] in valid, f"{name} has invalid category"

    def test_param_entries_have_type_default_required(self):
        for name, meta in INDICATOR_META.items():
            for pname, pdef in meta["params"].items():
                assert "type" in pdef, f"{name}.{pname} missing type"
                assert "default" in pdef, f"{name}.{pname} missing default"
                assert "required" in pdef, f"{name}.{pname} missing required"

    def test_param_types_are_valid(self):
        valid_types = {"int", "float", "str", "bool"}
        for name, meta in INDICATOR_META.items():
            for pname, pdef in meta["params"].items():
                assert pdef["type"] in valid_types, (
                    f"{name}.{pname} has invalid type {pdef['type']!r}"
                )

    def test_output_is_array_or_list(self):
        for name, meta in INDICATOR_META.items():
            out = meta["output"]
            assert out == "array" or isinstance(out, list), (
                f"{name} output should be 'array' or list of fields"
            )


class TestGetMeta:
    def test_returns_correct_data(self):
        meta = get_meta("rsi")
        assert meta["category"] == "momentum"
        assert meta["inputs"] == ["prices"]
        assert meta["params"]["period"]["default"] == 14

    def test_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown indicator"):
            get_meta("nonexistent")


class TestListBy:
    def test_list_by_category_trend(self):
        trend = list_by_category("trend")
        assert "sma" in trend
        assert "ema" in trend
        assert "efficiency_ratio" in trend
        assert "rsi" not in trend

    def test_list_by_category_empty(self):
        result = list_by_category("nonexistent")
        assert result == []

    def test_list_by_input_prices(self):
        prices_indicators = list_by_input("prices")
        assert "rsi" in prices_indicators
        assert "sma" in prices_indicators
        assert "volume_roc" not in prices_indicators

    def test_list_by_input_volume(self):
        vol = list_by_input("volume")
        assert set(vol) == {"volume_roc", "volume_zscore"}

    def test_list_by_input_high(self):
        high = list_by_input("high")
        assert "atr" in high
        assert "true_range" in high
        assert "stochastic" in high
        assert "rsi" not in high


class TestSchema:
    def test_schema_includes_name(self):
        s = schema("rsi")
        assert s["name"] == "rsi"
        assert s["category"] == "momentum"

    def test_schema_includes_params(self):
        s = schema("bollinger_bands")
        assert "period" in s["params"]
        assert s["params"]["period"]["type"] == "int"
        assert s["params"]["num_std"]["type"] == "float"

    def test_schema_includes_output_fields(self):
        s = schema("macd")
        assert s["output"] == ["macd_line", "signal", "histogram"]


class TestDefaults:
    def test_defaults_kama(self):
        d = defaults("kama")
        assert d == {"period": 10, "fast": 2, "slow": 30}

    def test_defaults_rsi(self):
        d = defaults("rsi")
        assert d == {"period": 14}

    def test_defaults_true_range_empty(self):
        d = defaults("true_range")
        assert d == {}

    def test_defaults_sma_empty(self):
        # sma has required period with no default
        d = defaults("sma")
        assert d == {}

    def test_defaults_ema_excludes_none(self):
        # alpha has default=None, should be excluded
        d = defaults("ema")
        assert "alpha" not in d


class TestRequiredParams:
    def test_sma_requires_period(self):
        assert required_params("sma") == ["period"]

    def test_wma_requires_period(self):
        assert required_params("wma") == ["period"]

    def test_ema_requires_period(self):
        assert "period" in required_params("ema")

    def test_rsi_no_required(self):
        assert required_params("rsi") == []

    def test_true_range_no_required(self):
        assert required_params("true_range") == []


class TestOutputFields:
    def test_single_output_returns_none(self):
        assert output_fields("rsi") is None
        assert output_fields("sma") is None
        assert output_fields("atr") is None

    def test_macd_fields(self):
        assert output_fields("macd") == ["macd_line", "signal", "histogram"]

    def test_bollinger_fields(self):
        assert output_fields("bollinger_bands") == [
            "middle", "upper", "lower", "bandwidth", "percent_b"
        ]

    def test_stochastic_fields(self):
        assert output_fields("stochastic") == ["k", "d"]

    def test_donchian_fields(self):
        assert output_fields("donchian_channels") == ["upper", "lower", "mid"]

    def test_linreg_fields(self):
        assert output_fields("linreg") == [
            "value", "slope", "intercept", "r_squared"
        ]


class TestValidateMeta:
    def test_validate_meta_passes(self):
        """Metadata must match actual function signatures."""
        validate_meta()
