"""
Kaufman Adaptive Trading System

Concept
-------
Adaptive system that switches behavior depending on market efficiency.

Efficiency Ratio (ER) determines regime:

High ER  → trending market → trend-following behavior
Low ER   → noisy market    → mean-reversion behavior

Trend Mode
----------
Price vs KAMA

Mean-Reversion Mode
-------------------
RSI reversal

Position sizing
---------------
ATR-normalized sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class KaufmanAdaptiveSystem(TradingSystem):

    def __init__(
        self,
        er_period: int = 10,
        trend_threshold: float = 0.40,
        noise_threshold: float = 0.20,
        fast_kama: int = 2,
        slow_kama: int = 30,
        rsi_period: int = 14,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.er_period = er_period
        self.trend_threshold = trend_threshold
        self.noise_threshold = noise_threshold
        self.fast_kama = fast_kama
        self.slow_kama = slow_kama
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    # ---------------------------------------------------------
    # Efficiency Ratio
    # ---------------------------------------------------------

    def efficiency_ratio(self, prices):

        prices = np.asarray(prices)

        if len(prices) < self.er_period + 1:
            return None

        change = abs(prices[-1] - prices[-1 - self.er_period])
        volatility = np.sum(np.abs(np.diff(prices[-1 - self.er_period :])))

        if volatility == 0:
            return 0

        return change / volatility

    # ---------------------------------------------------------
    # KAMA
    # ---------------------------------------------------------

    def kama(self, prices):

        prices = np.asarray(prices)

        if len(prices) < self.er_period + 1:
            return None

        er = self.efficiency_ratio(prices)

        if er is None:
            return None

        fast_sc = 2 / (self.fast_kama + 1)
        slow_sc = 2 / (self.slow_kama + 1)

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama_val = prices[-self.er_period]

        for price in prices[-self.er_period + 1 :]:
            kama_val = kama_val + sc * (price - kama_val)

        return kama_val

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    def rsi(self, prices):

        prices = np.asarray(prices)

        if len(prices) < self.rsi_period + 1:
            return None

        deltas = np.diff(prices[-(self.rsi_period + 1) :])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # ---------------------------------------------------------
    # ATR
    # ---------------------------------------------------------

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

        return np.mean(tr[-self.atr_period :])

    # ---------------------------------------------------------
    # Signal Logic
    # ---------------------------------------------------------

    def signal(self, data):

        closes = data["closes"]
        er = self.efficiency_ratio(closes)

        if er is None:
            return 0

        price = closes[-1]

        # TREND MODE
        if er >= self.trend_threshold:

            kama_val = self.kama(closes)

            if kama_val is None:
                return 0

            if price > kama_val:
                return 1
            elif price < kama_val:
                return -1
            else:
                return 0

        # NOISE MODE (mean reversion)
        if er <= self.noise_threshold:

            rsi_val = self.rsi(closes)

            if rsi_val is None:
                return 0

            if rsi_val < 30:
                return 1
            elif rsi_val > 70:
                return -1
            else:
                return 0

        # Neutral regime
        return 0

    # ---------------------------------------------------------
    # Position sizing
    # ---------------------------------------------------------

    def position_sizing(self, data, risk):

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)

        atr = self.atr(highs, lows, closes)

        if atr is None or atr == 0:
            return 0

        risk_amount = equity * risk_per_trade
        position = risk_amount / atr

        return position

    # ---------------------------------------------------------
    # Risk filter
    # ---------------------------------------------------------

    def risk_filter(self, data):

        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]

        atr = self.atr(highs, lows, closes)

        if atr is None:
            return False

        if atr <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def indicators(self, data):

        closes = data["closes"]

        return {
            "efficiency_ratio": self.efficiency_ratio(closes),
            "kama": self.kama(closes),
            "rsi": self.rsi(closes),
        }
