"""Streaming (stateful) indicator classes.

Each indicator maintains internal state and accepts new data one bar at a time
via :meth:`update`, making them suitable for:

- QuantConnect / LEAN bar-by-bar processing
- Real-time / live trading pipelines
- Any system that processes data incrementally

All streaming indicators share a common base class :class:`Indicator` that
provides a consistent interface: ``update()``, ``reset()``, ``value``,
and ``is_ready``.

Usage
-----
>>> from kaufman_indicators.streaming import StreamingEMA
>>> ema = StreamingEMA(period=10)
>>> for price in prices:
...     ema.update(price)
>>> ema.value   # current EMA value
>>> ema.is_ready  # True once enough data has been received

Reference
---------
Kaufman, P. J. (2013). *Trading Systems and Methods* (5th ed.).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any


class Indicator:
    """Base class for all streaming indicators.

    Subclasses must implement :meth:`update` and should set
    ``self._value`` when a new output is available.

    Attributes
    ----------
    name : str
        Human-readable name for the indicator instance.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self._value: float | None = None
        self._count: int = 0

    @property
    def value(self) -> float | None:
        """The current indicator value, or ``None`` if not yet ready."""
        return self._value

    @property
    def is_ready(self) -> bool:
        """Whether the indicator has received enough data to produce output."""
        return self._value is not None

    @property
    def count(self) -> int:
        """Number of samples received so far."""
        return self._count

    def update(self, price: float) -> float | None:
        """Feed a new data point and return the updated value.

        Parameters
        ----------
        price:
            The new price (or other scalar value) to incorporate.

        Returns
        -------
        float or None
            The updated indicator value, or ``None`` if not yet ready.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset all internal state so the indicator can be reused."""
        self._value = None
        self._count = 0

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready else "warming"
        return f"{self.name}({ready}, count={self._count})"


# ── Trend Indicators ─────────────────────────────────────────────────────────


class StreamingSMA(Indicator):
    """Streaming Simple Moving Average.

    Maintains a rolling window of the last *period* values and returns
    their arithmetic mean.
    """

    def __init__(self, period: int, name: str = "") -> None:
        super().__init__(name or f"SMA({period})")
        self.period = period
        self._buffer: deque[float] = deque(maxlen=period)
        self._sum: float = 0.0

    def update(self, price: float) -> float | None:
        self._count += 1
        if len(self._buffer) == self.period:
            self._sum -= self._buffer[0]
        self._buffer.append(price)
        self._sum += price
        if len(self._buffer) == self.period:
            self._value = self._sum / self.period
        return self._value

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._sum = 0.0


class StreamingEMA(Indicator):
    """Streaming Exponential Moving Average.

    Seeds from the SMA of the first *period* values, then applies
    exponential smoothing with ``alpha = 2 / (period + 1)``.
    """

    def __init__(self, period: int, alpha: float | None = None, name: str = "") -> None:
        super().__init__(name or f"EMA({period})")
        self.period = period
        self.alpha = alpha if alpha is not None else 2.0 / (period + 1)
        self._seed_values: list[float] = []

    def update(self, price: float) -> float | None:
        self._count += 1
        if self._value is not None:
            self._value = self._value + self.alpha * (price - self._value)
        else:
            self._seed_values.append(price)
            if len(self._seed_values) == self.period:
                self._value = sum(self._seed_values) / self.period
                self._seed_values = []
        return self._value

    def reset(self) -> None:
        super().reset()
        self._seed_values = []


class StreamingKAMA(Indicator):
    """Streaming Kaufman Adaptive Moving Average.

    Uses the Efficiency Ratio to adapt speed between a fast and slow
    smoothing constant.
    """

    def __init__(
        self,
        period: int = 10,
        fast: int = 2,
        slow: int = 30,
        name: str = "",
    ) -> None:
        super().__init__(name or f"KAMA({period},{fast},{slow})")
        self.period = period
        self._fast_sc = 2.0 / (fast + 1)
        self._slow_sc = 2.0 / (slow + 1)
        self._prices: deque[float] = deque(maxlen=period + 1)

    def update(self, price: float) -> float | None:
        self._count += 1
        self._prices.append(price)

        if len(self._prices) <= self.period:
            return self._value

        if self._value is None:
            self._value = price
            return self._value

        # Efficiency Ratio
        direction = abs(price - self._prices[0])
        volatility = sum(
            abs(self._prices[i] - self._prices[i - 1])
            for i in range(1, len(self._prices))
        )
        er = direction / volatility if volatility != 0 else 0.0

        sc = (er * (self._fast_sc - self._slow_sc) + self._slow_sc) ** 2
        self._value = self._value + sc * (price - self._value)
        return self._value

    def reset(self) -> None:
        super().reset()
        self._prices.clear()


# ── Momentum Indicators ─────────────────────────────────────────────────────


class StreamingROC(Indicator):
    """Streaming Rate of Change (percentage).

    ROC = (price / price_n_periods_ago - 1) * 100
    """

    def __init__(self, period: int = 12, name: str = "") -> None:
        super().__init__(name or f"ROC({period})")
        self.period = period
        self._buffer: deque[float] = deque(maxlen=period + 1)

    def update(self, price: float) -> float | None:
        self._count += 1
        self._buffer.append(price)
        if len(self._buffer) == self.period + 1:
            old = self._buffer[0]
            if old != 0:
                self._value = (price / old - 1.0) * 100.0
            else:
                self._value = 0.0
        return self._value

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()


class StreamingMomentum(Indicator):
    """Streaming Momentum (absolute price difference).

    Momentum = price - price_n_periods_ago
    """

    def __init__(self, period: int = 10, name: str = "") -> None:
        super().__init__(name or f"Momentum({period})")
        self.period = period
        self._buffer: deque[float] = deque(maxlen=period + 1)

    def update(self, price: float) -> float | None:
        self._count += 1
        self._buffer.append(price)
        if len(self._buffer) == self.period + 1:
            self._value = price - self._buffer[0]
        return self._value

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()


class StreamingRSI(Indicator):
    """Streaming Relative Strength Index using Wilder's smoothing.

    Seeds the average gain/loss from the first *period* price changes,
    then updates incrementally.
    """

    def __init__(self, period: int = 14, name: str = "") -> None:
        super().__init__(name or f"RSI({period})")
        self.period = period
        self._prev_price: float | None = None
        self._avg_gain: float = 0.0
        self._avg_loss: float = 0.0
        self._changes: list[float] = []

    def update(self, price: float) -> float | None:
        self._count += 1

        if self._prev_price is None:
            self._prev_price = price
            return self._value

        change = price - self._prev_price
        self._prev_price = price

        if self._value is None:
            # Still seeding
            self._changes.append(change)
            if len(self._changes) == self.period:
                gains = [c for c in self._changes if c > 0]
                losses = [-c for c in self._changes if c < 0]
                self._avg_gain = sum(gains) / self.period
                self._avg_loss = sum(losses) / self.period
                self._changes = []
                self._value = self._compute_rsi()
        else:
            # Incremental update
            gain = change if change > 0 else 0.0
            loss = -change if change < 0 else 0.0
            alpha = 1.0 / self.period
            self._avg_gain = self._avg_gain * (1.0 - alpha) + gain * alpha
            self._avg_loss = self._avg_loss * (1.0 - alpha) + loss * alpha
            self._value = self._compute_rsi()

        return self._value

    def _compute_rsi(self) -> float:
        if self._avg_loss == 0.0:
            return 100.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def reset(self) -> None:
        super().reset()
        self._prev_price = None
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._changes = []


class StreamingMACD(Indicator):
    """Streaming MACD (Moving Average Convergence/Divergence).

    Internally composes three :class:`StreamingEMA` instances.

    The ``value`` property returns the MACD line. Use ``signal_value``
    and ``histogram_value`` for the other components.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        name: str = "",
    ) -> None:
        super().__init__(name or f"MACD({fast},{slow},{signal_period})")
        self._fast_ema = StreamingEMA(fast)
        self._slow_ema = StreamingEMA(slow)
        self._signal_ema = StreamingEMA(signal_period)
        self._signal_value: float | None = None
        self._histogram_value: float | None = None

    @property
    def signal_value(self) -> float | None:
        """Current signal line value."""
        return self._signal_value

    @property
    def histogram_value(self) -> float | None:
        """Current histogram value (MACD line - signal)."""
        return self._histogram_value

    def update(self, price: float) -> float | None:
        self._count += 1
        fast_val = self._fast_ema.update(price)
        slow_val = self._slow_ema.update(price)

        if fast_val is not None and slow_val is not None:
            self._value = fast_val - slow_val
            sig = self._signal_ema.update(self._value)
            if sig is not None:
                self._signal_value = sig
                self._histogram_value = self._value - sig

        return self._value

    def reset(self) -> None:
        super().reset()
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()
        self._signal_value = None
        self._histogram_value = None


# ── Volatility Indicators ────────────────────────────────────────────────────


class StreamingATR(Indicator):
    """Streaming Average True Range using Wilder's smoothing.

    Requires high, low, close per bar via :meth:`update`.
    """

    def __init__(self, period: int = 14, name: str = "") -> None:
        super().__init__(name or f"ATR({period})")
        self.period = period
        self._prev_close: float | None = None
        self._tr_values: list[float] = []

    def update(self, price: float, high: float | None = None, low: float | None = None) -> float | None:
        """Update with a new bar.

        Parameters
        ----------
        price:
            Close price for the bar.
        high:
            High price for the bar. If ``None``, uses *price*.
        low:
            Low price for the bar. If ``None``, uses *price*.
        """
        self._count += 1
        if high is None:
            high = price
        if low is None:
            low = price

        # True Range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        self._prev_close = price

        if self._value is None:
            self._tr_values.append(tr)
            if len(self._tr_values) == self.period:
                self._value = sum(self._tr_values) / self.period
                self._tr_values = []
        else:
            alpha = 1.0 / self.period
            self._value = self._value * (1.0 - alpha) + tr * alpha

        return self._value

    def reset(self) -> None:
        super().reset()
        self._prev_close = None
        self._tr_values = []


# ── Range Indicators ─────────────────────────────────────────────────────────


class StreamingBollingerBands(Indicator):
    """Streaming Bollinger Bands.

    The ``value`` property returns the middle band (SMA). Use
    ``upper``, ``lower``, ``bandwidth``, and ``percent_b`` for the
    other components.
    """

    def __init__(self, period: int = 20, num_std: float = 2.0, name: str = "") -> None:
        super().__init__(name or f"BB({period},{num_std})")
        self.period = period
        self.num_std = num_std
        self._buffer: deque[float] = deque(maxlen=period)
        self._upper: float | None = None
        self._lower: float | None = None
        self._bandwidth: float | None = None
        self._percent_b: float | None = None

    @property
    def upper(self) -> float | None:
        """Upper Bollinger Band value."""
        return self._upper

    @property
    def lower(self) -> float | None:
        """Lower Bollinger Band value."""
        return self._lower

    @property
    def bandwidth(self) -> float | None:
        """Bandwidth as a percentage of the middle band."""
        return self._bandwidth

    @property
    def percent_b(self) -> float | None:
        """Position of the last price within the bands (0..1)."""
        return self._percent_b

    def update(self, price: float) -> float | None:
        self._count += 1
        self._buffer.append(price)

        if len(self._buffer) == self.period:
            values = list(self._buffer)
            mean = sum(values) / self.period
            variance = sum((x - mean) ** 2 for x in values) / (self.period - 1)
            std = math.sqrt(variance)

            self._value = mean
            self._upper = mean + self.num_std * std
            self._lower = mean - self.num_std * std

            if mean != 0:
                self._bandwidth = (self._upper - self._lower) / mean * 100.0
            else:
                self._bandwidth = None

            band_range = self._upper - self._lower
            if band_range != 0:
                self._percent_b = (price - self._lower) / band_range
            else:
                self._percent_b = None

        return self._value

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._upper = None
        self._lower = None
        self._bandwidth = None
        self._percent_b = None


# ── Factory ──────────────────────────────────────────────────────────────────


STREAMING_INDICATORS: dict[str, type[Indicator]] = {
    "sma": StreamingSMA,
    "ema": StreamingEMA,
    "kama": StreamingKAMA,
    "roc": StreamingROC,
    "momentum": StreamingMomentum,
    "rsi": StreamingRSI,
    "macd": StreamingMACD,
    "atr": StreamingATR,
    "bollinger_bands": StreamingBollingerBands,
}


def create_streaming(name: str, **kwargs: Any) -> Indicator:
    """Create a streaming indicator by name.

    Parameters
    ----------
    name:
        Indicator name (e.g. ``"rsi"``, ``"ema"``).
    **kwargs:
        Parameters forwarded to the indicator constructor.

    Returns
    -------
    Indicator
        A new streaming indicator instance.

    Raises
    ------
    KeyError
        If *name* is not a registered streaming indicator.

    Examples
    --------
    >>> ind = create_streaming("rsi", period=14)
    >>> ind.update(100.0)
    >>> ind.update(101.5)
    """
    try:
        cls = STREAMING_INDICATORS[name]
    except KeyError:
        available = ", ".join(sorted(STREAMING_INDICATORS))
        raise KeyError(
            f"Unknown streaming indicator {name!r}. Available: {available}"
        ) from None
    return cls(**kwargs)
