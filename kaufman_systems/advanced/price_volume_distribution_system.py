"""
Price Distribution System

Identifies the value area from time-at-price distribution.
Bins recent closes into price levels and finds where the most
time (bars) was spent.  Breakouts above/below the value area
generate signals.

Since we work with closes/highs/lows only (no volume data),
we use a time-at-price proxy: the number of bars whose close
falls into each price bin.

Corresponds to Kaufman Chapter 18 — Price Distribution Systems.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PriceVolumeDistributionSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 50,
        num_bins: int = 20,
        value_area_pct: float = 0.70,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.lookback = lookback
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def value_area(self, closes):
        closes = np.asarray(closes)

        if len(closes) < self.lookback + 1:
            return None, None

        window = closes[-(self.lookback + 1):-1]
        price_min = window.min()
        price_max = window.max()

        if price_min == price_max:
            return None, None

        bin_edges = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_indices = np.digitize(window, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        bin_counts = np.zeros(self.num_bins)
        for idx in bin_indices:
            bin_counts[idx] += 1

        total = bin_counts.sum()
        if total == 0:
            return None, None

        sorted_idx = np.argsort(bin_counts)[::-1]
        cumulative = 0
        value_bins = []

        for idx in sorted_idx:
            cumulative += bin_counts[idx]
            value_bins.append(idx)
            if cumulative >= total * self.value_area_pct:
                break

        value_bins = sorted(value_bins)
        va_low = bin_edges[value_bins[0]]
        va_high = bin_edges[value_bins[-1] + 1]

        return va_low, va_high

    def atr(self, highs, lows, closes):
        highs = np.asarray(highs)
        lows = np.asarray(lows)
        closes = np.asarray(closes)

        if len(closes) < self.atr_period + 1:
            return None

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        return np.mean(tr[-self.atr_period:])

    def signal(self, data):
        closes = np.asarray(data["closes"])

        va_low, va_high = self.value_area(closes)

        if va_low is None:
            return 0

        if closes[-1] > va_high:
            return 1
        elif closes[-1] < va_low:
            return -1
        return 0

    def position_sizing(self, data, risk):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr_val = self.atr(highs, lows, closes)

        if atr_val is None or atr_val == 0:
            return 0

        return equity * risk_per_trade / atr_val

    def risk_filter(self, data):
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr_val = self.atr(highs, lows, closes)

        if atr_val is None or atr_val <= 0:
            return False
        return True

    def indicators(self, data):
        closes = data["closes"]
        va_low, va_high = self.value_area(closes)
        return {
            "value_area_low": va_low,
            "value_area_high": va_high,
        }
