"""
Fisheries stock, price and risk models.
Mini-Project 3: logistic growth with harvesting, random-walk pricing, VaR.
"""
from __future__ import annotations

import numpy as np
from statistics import mean, median, variance, stdev


class FishStock:
    """
    Discrete logistic growth with proportional harvesting.

    Recurrence:
        N(t+1) = N(t) + r * N(t) * (1 - N(t)/K) - h * N(t)

    Parameters
    ----------
    r : float
        Intrinsic growth rate per week.
    K : float
        Carrying capacity (tonnes).
    N0 : float
        Initial stock (tonnes).
    h : float
        Harvest rate (fraction of stock removed per week).
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0,
                 N0: float = 4_000.0, h: float = 0.10):
        if r < 0 or K <= 0 or N0 < 0 or h < 0:
            raise ValueError("r, K, N0, h must be non-negative (K > 0)")
        if N0 > K:
            raise ValueError("N0 cannot exceed carrying capacity K")
        self.r = float(r)
        self.K = float(K)
        self.N0 = float(N0)
        self.h = float(h)

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Return array of length weeks+1 (index 0 = N0)."""
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        N = np.empty(weeks + 1)
        N[0] = self.N0
        for t in range(weeks):
            growth = self.r * N[t] * (1 - N[t] / self.K)
            harvest = self.h * N[t]
            N[t + 1] = max(0.0, N[t] + growth - harvest)
        return N

    def msy(self) -> float:
        """Maximum sustainable yield (tonnes/week): r*K/4."""
        return self.r * self.K / 4


class PriceModel:
    """
    Seeded random-walk price simulation with reflecting bounds.

    Price starts at `start` and evolves as  P[t] = P[t-1] + N(0, sigma),
    clipped to [low, high].
    """

    def __init__(self, start: float = 12_000.0, low: float = 9_000.0,
                 high: float = 16_000.0, sigma: float = 300.0, seed: int = 42):
        if low >= high:
            raise ValueError("low must be < high")
        if not (low <= start <= high):
            raise ValueError("start must lie within [low, high]")
        self.start = float(start)
        self.low = float(low)
        self.high = float(high)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def simulate(self, weeks: int = 52) -> np.ndarray:
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        p = np.empty(weeks)
        p[0] = self.start
        for t in range(1, weeks):
            step = self.rng.normal(0, self.sigma)
            p[t] = np.clip(p[t - 1] + step, self.low, self.high)
        return p

    def simulate_paths(self, weeks: int, n_paths: int) -> np.ndarray:
        """Return (n_paths, weeks) array of price paths."""
        out = np.empty((n_paths, weeks))
        for i in range(n_paths):
            out[i] = self.simulate(weeks)
        return out


class RiskAssessor:
    """
    Classifies revenue risk based on the coefficient of variation (CV).

    Defaults: CV < 0.10 → Low; < 0.25 → Medium; else High.
    """

    def __init__(self, low_cv: float = 0.10, high_cv: float = 0.25):
        if low_cv <= 0 or high_cv <= low_cv:
            raise ValueError("Require 0 < low_cv < high_cv")
        self.low_cv = float(low_cv)
        self.high_cv = float(high_cv)

    def cv(self, revenues: np.ndarray) -> float:
        arr = np.asarray(revenues, dtype=float)
        m = arr.mean()
        return float(arr.std(ddof=1) / m) if m > 0 else float("nan")

    def classify(self, revenues: np.ndarray) -> str:
        cv = self.cv(revenues)
        if cv < self.low_cv:
            return "Low"
        if cv < self.high_cv:
            return "Medium"
        return "High"

    def value_at_risk(self, revenues: np.ndarray, alpha: float = 0.05) -> float:
        """Return the alpha-quantile of the revenue distribution."""
        return float(np.percentile(revenues, alpha * 100))