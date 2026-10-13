"""
Module for district population forecasting.
Mini-Project 1: UBOS District Population Forecaster.

Pure logic only — no plots, no I/O. The Jupyter notebook imports from here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class DistrictPopulation:
    """
    Stores a district's yearly population estimates.

    Attributes
    ----------
    name : str
        District name.
    years : np.ndarray
        1-D integer array of years.
    populations : np.ndarray
        1-D float array of populations (thousands), same length as years.
    """

    def __init__(self, name: str, years: Iterable[int], populations: Iterable[float]):
        self.name = name
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)

        if self.years.ndim != 1 or self.populations.ndim != 1:
            raise ValueError("years and populations must be 1-D sequences")
        if len(self.years) != len(self.populations):
            raise ValueError("years and populations must have equal length")
        if len(self.years) == 0:
            raise ValueError("Inputs must not be empty")
        if np.any(self.populations < 0):
            raise ValueError("Populations cannot be negative")
        if np.any(np.diff(self.years) <= 0):
            raise ValueError("Years must be strictly increasing")

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, "
            f"years={self.years[0]}-{self.years[-1]}, n={len(self)})"
        )

    def __len__(self) -> int:
        return len(self.years)

    def growth_rates(self) -> np.ndarray:
        """Year-on-year growth rate: (P[t] - P[t-1]) / P[t-1]."""
        p = self.populations
        return (p[1:] - p[:-1]) / p[:-1]

    def cagr(self) -> float:
        """Compound Annual Growth Rate from first to last year."""
        n = len(self.populations) - 1
        if n <= 0:
            return 0.0
        first, last = self.populations[0], self.populations[-1]
        if first <= 0:
            return 0.0
        return float((last / first) ** (1 / n) - 1)


class Forecaster(ABC):
    """Abstract base class for population forecasters."""

    def __init__(self) -> None:
        self._fitted = False
        self._last_year: int = 0
        self._last_value: float = 0.0

    @abstractmethod
    def fit(self, years: np.ndarray, values: np.ndarray) -> "Forecaster":
        """Fit the model to training data."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return `horizon` future values."""


class LinearTrendForecaster(Forecaster):
    """Fits value ~ slope * year + intercept via least squares."""

    def __init__(self) -> None:
        super().__init__()
        self.slope: float = 0.0
        self.intercept: float = 0.0

    def fit(self, years: np.ndarray, values: np.ndarray) -> "LinearTrendForecaster":
        self.slope, self.intercept = np.polyfit(years, values, 1)
        self._last_year = int(years[-1])
        self._last_value = float(values[-1])
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        future_years = np.arange(self._last_year + 1, self._last_year + 1 + horizon)
        return self.slope * future_years + self.intercept


class CAGRForecaster(Forecaster):
    """Compounds the last observed value at the historical CAGR."""

    def __init__(self) -> None:
        super().__init__()
        self.rate: float = 0.0

    def fit(self, years: np.ndarray, values: np.ndarray) -> "CAGRForecaster":
        n = len(values) - 1
        self.rate = (
            (values[-1] / values[0]) ** (1 / n) - 1
            if n > 0 and values[0] > 0
            else 0.0
        )
        self._last_year = int(years[-1])
        self._last_value = float(values[-1])
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        steps = np.arange(1, horizon + 1)
        return self._last_value * (1 + self.rate) ** steps


class FibonacciForecaster(Forecaster):
    """
    Scales the last value by successive Fibonacci ratios.

    Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13, ...
    Ratio of consecutive terms approaches phi ≈ 1.618.
    We use F(k+1)/F(k) as a multiplicative growth factor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fib = self._fibonacci(20)

    @staticmethod
    def _fibonacci(n: int) -> np.ndarray:
        seq = np.zeros(n)
        if n >= 1:
            seq[0] = 1
        if n >= 2:
            seq[1] = 1
        for i in range(2, n):
            seq[i] = seq[i - 1] + seq[i - 2]
        return seq

    def fit(self, years: np.ndarray, values: np.ndarray) -> "FibonacciForecaster":
        self._last_year = int(years[-1])
        self._last_value = float(values[-1])
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        ratios = self._fib[2 : 2 + horizon] / self._fib[1 : 1 + horizon]
        out = np.empty(horizon)
        current = self._last_value
        for i, r in enumerate(ratios):
            current = current * r
            out[i] = current
        return out


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(diff ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error (percentage, e.g. 5.2 for 5.2%)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)