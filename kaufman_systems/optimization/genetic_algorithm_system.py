"""
Genetic Algorithm System

Concept
-------
Evolves the (fast, slow) moving-average crossover parameters on the recent
price window with a small seeded genetic algorithm (tournament selection,
uniform crossover, mutation, elitism).  The fittest genome defines the
current crossover signal.

Signal logic
------------
Fitness = in-sample return of a long/short MA crossover over the last
``train_window`` bars.  With the best (fast, slow):
  fast MA > slow MA → LONG
  fast MA < slow MA → SHORT
Otherwise           → FLAT

The RNG is seeded, so results are deterministic.

Risk model
----------
ATR-based position sizing.
"""

import numpy as np

from kaufman_systems.base import TradingSystem


class GeneticAlgorithmSystem(TradingSystem):

    def __init__(
        self,
        fast_range: tuple = (3, 20),
        slow_range: tuple = (21, 50),
        train_window: int = 250,
        population: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        seed: int = 42,
        atr_period: int = 14,
        risk_per_trade: float = 0.01,
    ):
        self.fast_range = fast_range
        self.slow_range = slow_range
        self.train_window = train_window
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    @property
    def min_bars(self):
        return self.slow_range[1] + 20

    # ---------------------------------------------------------
    # Indicators
    # ---------------------------------------------------------

    @staticmethod
    def _sma(x, n):
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[n:] - c[:-n]) / n

    def _fitness(self, closes, fast, slow):
        fast_ma = self._sma(closes, fast)[slow - fast:]
        slow_ma = self._sma(closes, slow)
        pos = np.sign(fast_ma - slow_ma)
        rets = np.diff(closes[slow - 1:]) / closes[slow - 1:-1]
        return float(np.sum(pos[:-1] * rets))

    def evolve(self, closes):
        closes = np.asarray(closes, dtype=float)[-self.train_window:]
        rng = np.random.default_rng(self.seed)
        f_lo, f_hi = self.fast_range
        s_lo, s_hi = self.slow_range

        pop = np.column_stack([
            rng.integers(f_lo, f_hi + 1, self.population),
            rng.integers(s_lo, s_hi + 1, self.population),
        ])
        best, best_fit = None, -np.inf

        for _ in range(self.generations):
            fit = np.array([self._fitness(closes, f, s) for f, s in pop])
            top = int(np.argmax(fit))
            if fit[top] > best_fit:
                best_fit, best = float(fit[top]), tuple(int(v) for v in pop[top])

            children = [pop[top]]
            while len(children) < self.population:
                a = max(rng.integers(0, self.population, 2), key=lambda i: fit[i])
                b = max(rng.integers(0, self.population, 2), key=lambda i: fit[i])
                child = np.where(rng.random(2) < 0.5, pop[a], pop[b])
                if rng.random() < self.mutation_rate:
                    child[0] = rng.integers(f_lo, f_hi + 1)
                if rng.random() < self.mutation_rate:
                    child[1] = rng.integers(s_lo, s_hi + 1)
                children.append(child)
            pop = np.array(children)

        return best, best_fit

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
        closes = np.asarray(data["closes"], dtype=float)
        if len(closes) < self.min_bars:
            return 0

        (fast, slow), _ = self.evolve(closes)
        diff = np.mean(closes[-fast:]) - np.mean(closes[-slow:])

        if diff > 0:
            return 1
        if diff < 0:
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
        closes = np.asarray(data["closes"], dtype=float)
        if len(closes) < self.min_bars:
            return {"best_fast": None, "best_slow": None, "best_fitness": None}
        (fast, slow), fit = self.evolve(closes)
        return {"best_fast": fast, "best_slow": slow, "best_fitness": fit}
