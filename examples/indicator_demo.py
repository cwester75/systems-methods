"""indicator_demo.py – Quick demonstration of all kaufman_indicators modules.

Run from the repository root::

    pip install -r requirements.txt
    python examples/indicator_demo.py
"""

import numpy as np

import kaufman_indicators as ki

# ── Synthetic price data ────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n = 200
close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
spread = np.abs(rng.standard_normal(n)) * 0.3 + 0.2
high = close + spread
low = close - spread


def _last(arr: np.ndarray) -> str:
    """Return the last non-NaN value as a formatted string."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return "N/A"
    return f"{valid[-1]:.4f}"


# ── Trend ───────────────────────────────────────────────────────────────────
print("=== Trend ===")
er = ki.efficiency_ratio(close, period=10)
print(f"  Efficiency Ratio (10)  : {_last(er)}")

kama_vals = ki.kama(close, period=10, fast=2, slow=30)
print(f"  KAMA (10, 2, 30)       : {_last(kama_vals)}")

lr = ki.linreg(close, period=14)
print(f"  LinReg value (14)      : {_last(lr.value)}")
print(f"  LinReg slope (14)      : {_last(lr.slope)}")

print(f"  SMA (20)               : {_last(ki.sma(close, 20))}")
print(f"  EMA (20)               : {_last(ki.ema(close, 20))}")
print(f"  WMA (20)               : {_last(ki.wma(close, 20))}")
print(f"  DEMA (20)              : {_last(ki.dema(close, 20))}")
print(f"  TEMA (20)              : {_last(ki.tema(close, 20))}")

# ── Momentum ────────────────────────────────────────────────────────────────
print("\n=== Momentum ===")
print(f"  ROC (12)               : {_last(ki.roc(close, 12))}")
print(f"  RSI (14)               : {_last(ki.rsi(close, 14))}")

macd_result = ki.macd(close)
print(f"  MACD line              : {_last(macd_result.macd_line)}")
print(f"  MACD signal            : {_last(macd_result.signal)}")
print(f"  MACD histogram         : {_last(macd_result.histogram)}")

stoch = ki.stochastic(high, low, close)
print(f"  Stochastic %K (14)     : {_last(stoch.k)}")
print(f"  Stochastic %D (3)      : {_last(stoch.d)}")

# ── Volatility ──────────────────────────────────────────────────────────────
print("\n=== Volatility ===")
tr = ki.true_range(high, low, close)
print(f"  True Range (last)      : {_last(tr)}")
print(f"  ATR (14)               : {_last(ki.atr(high, low, close, 14))}")
print(f"  Realized Vol (20, ann) : {_last(ki.realized_vol(close, 20))}")

# ── Range ───────────────────────────────────────────────────────────────────
print("\n=== Range ===")
bb = ki.bollinger_bands(close, period=20, num_std=2.0)
print(f"  Bollinger Upper (20,2) : {_last(bb.upper)}")
print(f"  Bollinger Middle       : {_last(bb.middle)}")
print(f"  Bollinger Lower        : {_last(bb.lower)}")
print(f"  %B                     : {_last(bb.percent_b)}")

dc = ki.donchian_channels(high, low, period=20)
print(f"  Donchian Upper (20)    : {_last(dc.upper)}")
print(f"  Donchian Lower (20)    : {_last(dc.lower)}")
print(f"  Donchian Mid           : {_last(dc.mid)}")

wr = ki.williams_r(high, low, close, period=14)
print(f"  Williams %R (14)       : {_last(wr)}")

# ── Market Quality ──────────────────────────────────────────────────────────
print("\n=== Market Quality ===")
print(f"  FDI (30)               : {_last(ki.fdi(close, 30))}")
print(f"  Hurst Exponent (100)   : {_last(ki.hurst_exponent(close, 100))}")
print(f"  Entropy Shannon (50)   : {_last(ki.price_entropy(close, 50, 'shannon'))}")
print(f"  Entropy ApEn (50)      : {_last(ki.price_entropy(close, 50, 'approximate'))}")

print("\nDemo complete.")
