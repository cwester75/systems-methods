"""
Neural Network Predictor

Concept
-------
A tiny feed-forward network (one tanh hidden layer) is trained with
full-batch gradient descent on lagged, volatility-normalised returns to
predict the next bar's direction.  Weights come from a seeded RNG and the
network is retrained on every call, so results are deterministic.

Signal logic
------------
prediction > +threshold → LONG
prediction < -threshold → SHORT
Otherwise               → FLAT

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class NeuralNetworkPredictor(TradingSystem):

    def __init__(
        self,
        n_lags: int = 5,
        hidden: int = 4,
        train_window: int = 120,
        epochs: int = 150,
        learning_rate: float = 0.05,
        threshold: float = 0.1,
        seed: int = 42,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.n_lags = n_lags
        self.hidden = hidden
        self.train_window = train_window
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.seed = seed
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    @property
    def min_bars(self):
        return self.n_lags + 31

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    def predict(self, closes):
        """Train on the recent window and return the next-bar prediction
        in (-1, 1), or None if there is not enough usable data."""
        closes = np.asarray(closes, dtype=float)[-self.train_window - 1:]
        if len(closes) < self.min_bars or np.any(closes <= 0):
            return None

        rets = np.diff(np.log(closes))
        sd = np.std(rets)
        if sd == 0:
            return None
        z = rets / sd

        n = len(z) - self.n_lags
        X = np.array([z[i:i + self.n_lags] for i in range(n)])
        y = np.tanh(z[self.n_lags:])
        x_new = z[-self.n_lags:]

        rng = np.random.default_rng(self.seed)
        W1 = rng.normal(0, 0.5, (self.n_lags, self.hidden))
        b1 = np.zeros(self.hidden)
        W2 = rng.normal(0, 0.5, self.hidden)
        b2 = 0.0

        for _ in range(self.epochs):
            h = np.tanh(X @ W1 + b1)
            out = np.tanh(h @ W2 + b2)
            d_out = (out - y) * (1 - out ** 2) / n
            gW2 = h.T @ d_out
            gb2 = d_out.sum()
            d_h = np.outer(d_out, W2) * (1 - h ** 2)
            gW1 = X.T @ d_h
            gb1 = d_h.sum(axis=0)
            W1 -= self.learning_rate * gW1
            b1 -= self.learning_rate * gb1
            W2 -= self.learning_rate * gW2
            b2 -= self.learning_rate * gb2

        return float(np.tanh(np.tanh(x_new @ W1 + b1) @ W2 + b2))

    def atr(self, highs, lows, closes):
        highs, lows, closes = np.asarray(highs), np.asarray(lows), np.asarray(closes)
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

    # ---------------------------------------------------------
    # Core interface
    # ---------------------------------------------------------

    def signal(self, data):
        pred = self.predict(data["closes"])
        if pred is None:
            return 0
        if pred > self.threshold:
            return 1
        if pred < -self.threshold:
            return -1
        return 0

    def position_sizing(self, data, risk):
        equity = risk["equity"]
        risk_per_trade = risk.get("risk_per_trade", self.risk_per_trade)
        atr = self.atr(data["highs"], data["lows"], data["closes"])
        if atr is None or atr == 0:
            return 0
        return equity * risk_per_trade / atr

    def risk_filter(self, data):
        atr = self.atr(data["highs"], data["lows"], data["closes"])
        return bool(atr is not None and atr > 0)

    def indicators(self, data):
        return {"prediction": self.predict(data["closes"])}
