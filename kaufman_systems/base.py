"""
TradingSystem — Abstract base class for all Kaufman trading systems.

Every system module exposes the same interface:

    signal(data)            → 1 (long), -1 (short), 0 (flat)
    position_sizing(data, risk) → position size / leverage
    risk_filter(data)       → True if trade is allowed
    indicators(data)        → dict of internal indicator values

``data`` is a dict containing price arrays.  Required keys depend on the
system, but the canonical set is::

    {
        "closes": np.ndarray,
        "highs":  np.ndarray,   # optional for some systems
        "lows":   np.ndarray,   # optional for some systems
    }

``risk`` is a dict describing account-level risk parameters::

    {
        "equity":         float,
        "risk_per_trade": float,   # optional — falls back to system default
    }

Standardising this interface enables lean adapters, research notebooks,
comparison testing, and UP spine integration.
"""

from abc import ABC, abstractmethod


class TradingSystem(ABC):
    """Abstract base for every trading system in the library."""

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def signal(self, data: dict) -> int:
        """Return trading signal.

        Returns
        -------
        1  → long
       -1  → short
        0  → flat / no signal
        """

    @abstractmethod
    def position_sizing(self, data: dict, risk: dict) -> float:
        """Return position size or leverage.

        Parameters
        ----------
        data : dict
            Price arrays (closes, highs, lows).
        risk : dict
            Must contain ``equity``.  May contain ``risk_per_trade``
            to override the system default.
        """

    @abstractmethod
    def risk_filter(self, data: dict) -> bool:
        """Return True if the trade passes risk checks."""

    # ------------------------------------------------------------------
    # Optional (but recommended)
    # ------------------------------------------------------------------

    def indicators(self, data: dict) -> dict:
        """Return dict of internal indicators for diagnostics."""
        return {}
