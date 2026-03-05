"""
kaufman_indicators – Technical analysis indicators based on
Perry Kaufman's "Trading Systems and Methods".

Subpackages
-----------
trend           : Efficiency Ratio, KAMA, Linear Regression, Moving Averages
momentum        : Rate of Change, RSI, MACD, Stochastic
volatility      : True Range, ATR, Realized Volatility
range           : Bollinger Bands, Donchian Channels, Williams %R
market_quality  : Fractal Dimension Index, Hurst Exponent, Entropy
utils           : Rolling helpers, Math helpers
"""

from kaufman_indicators.trend.efficiency_ratio import efficiency_ratio
from kaufman_indicators.trend.kama import kama
from kaufman_indicators.trend.linreg import linreg, linreg_forecast
from kaufman_indicators.trend.moving_averages import sma, ema, wma, dema, tema

from kaufman_indicators.momentum.roc import roc
from kaufman_indicators.momentum.rsi import rsi
from kaufman_indicators.momentum.macd import macd
from kaufman_indicators.momentum.stochastic import stochastic

from kaufman_indicators.volatility.true_range import true_range
from kaufman_indicators.volatility.atr import atr
from kaufman_indicators.volatility.realized_vol import realized_vol

from kaufman_indicators.range.bollinger import bollinger_bands
from kaufman_indicators.range.donchian import donchian_channels
from kaufman_indicators.range.williams_r import williams_r

from kaufman_indicators.market_quality.fdi import fdi
from kaufman_indicators.market_quality.hurst import hurst_exponent
from kaufman_indicators.market_quality.entropy import price_entropy

__all__ = [
    # trend
    "efficiency_ratio",
    "kama",
    "linreg", "linreg_forecast",
    "sma", "ema", "wma", "dema", "tema",
    # momentum
    "roc",
    "rsi",
    "macd",
    "stochastic",
    # volatility
    "true_range",
    "atr",
    "realized_vol",
    # range
    "bollinger_bands",
    "donchian_channels",
    "williams_r",
    # market_quality
    "fdi",
    "hurst_exponent",
    "price_entropy",
]
