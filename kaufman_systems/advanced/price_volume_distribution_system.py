"""
Price Volume Distribution System

Identifies value area from volume-weighted price distribution.
Breakouts above/below the value area generate signals.

Corresponds to Kaufman Chapter 18 — Price Distribution Systems.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class PriceVolumeDistributionSystem(TradingSystem):

    def __init__(
        self,
        lookback: int = 30,
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

    def value_area(self, closes, volumes):
        closes = np.asarray(closes[-self.lookback :])
        volumes = np.asarray(volumes[-self.lookback :])

        if len(closes) < self.lookback:
            return None, None

        bin_edges = np.linspace(closes.min(), closes.max(), self.num_bins + 1)
        bin_indices = np.digitize(closes, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        bin_volumes = np.zeros(self.num_bins)
        for i in range(len(closes)):
            bin_volumes[bin_indices[i]] += volumes[i]

        total_volume = bin_volumes.sum()
        if total_volume == 0:
            return None, None

        sorted_idx = np.argsort(bin_volumes)[::-1]
        cumulative = 0
        value_bins = []

        for idx in sorted_idx:
            cumulative += bin_volumes[idx]
            value_bins.append(idx)
            if cumulative >= total_volume * self.value_area_pct:
                break

        value_bins = sorted(value_bins)
        va_low = bin_edges[value_bins[0]]
        va_high = bin_edges[value_bins[-1] + 1]

        return va_low, va_high

    def signal(self, data):
        closes = np.asarray(data["closes"])
        volumes = np.asarray(data["volumes"])

        va_low, va_high = self.value_area(closes, volumes)

        if va_low is None:
            return 0

        if closes[-1] > va_high:
            return 1
        elif closes[-1] < va_low:
            return -1
        return 0

    def position_sizing(self, data, risk):
        closes = np.asarray(data["closes"])
        highs = np.asarray(data["highs"])
        lows = np.asarray(data["lows"])
        equity = risk["equity"]

        if len(closes) < self.atr_period + 1:
            return 0

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = np.mean(tr[-self.atr_period :])

        if atr == 0:
            return 0

        return equity * self.risk_per_trade / atr

    def risk_filter(self, data):
        closes = np.asarray(data["closes"])
        volumes = np.asarray(data["volumes"])

        va_low, va_high = self.value_area(closes, volumes)
        return va_low is not None

    def indicators(self, data):
        closes = data["closes"]
        volumes = data["volumes"]
        va_low, va_high = self.value_area(closes, volumes)
        return {
            "value_area_low": va_low,
            "value_area_high": va_high,
        }
