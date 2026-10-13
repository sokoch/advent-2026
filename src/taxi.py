"""
Taxi route revenue, pricing and fleet planning.
Mini-Project 5: Route statistics, supply-demand equilibrium, forecasters, fleet sizing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from scipy import linalg
from statistics import mean, variance, stdev


class Route:
    """
    A single taxi route: daily passenger counts and a flat fare.

    Parameters
    ----------
    name : str
    passengers : array-like
        Daily passenger counts (>= 0).
    fare : float
        Fare per passenger in UGX.
    """

    def __init__(self, name: str, passengers: Iterable[float], fare: float):
        self.name = name
        self.passengers = np.asarray(passengers, dtype=float)
        self.fare = float(fare)
        if self.passengers.ndim != 1 or len(self.passengers) == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if np.any(self.passengers < 0):
            raise ValueError("passenger counts cannot be negative")
        if self.fare < 0:
            raise ValueError("fare cannot be negative")

    def __repr__(self) -> str:
        return f"Route(name={self.name!r}, days={len(self.passengers)}, fare={self.fare:.0f})"

    def __len__(self) -> int:
        return len(self.passengers)

    def daily_revenue(self) -> np.ndarray:
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def stats(self) -> dict:
        rev = self.daily_revenue()
        return {
            "mean":     float(mean(rev)),
            "variance": float(variance(rev)),
            "stdev":    float(stdev(rev)),
        }


# ---------- Supply-demand equilibrium ----------
def solve_equilibrium(qd_intercept: float = 120.0, qd_slope: float = -0.02,
                      qs_intercept: float = 10.0,  qs_slope: float = 0.03) -> tuple[float, float]:
    """
    Solve  Qd = a - b·P  and  Qs = c + d·P  simultaneously.

    Rewrite:
        Q + b·P = a
        Q − d·P = c
    Matrix form:
        [[1,  b],   [[Q],   [[a],
         [1, -d]] ·  [P]] =  [c]]

    Returns (Q_eq, P_eq).
    """
    A = np.array([[1.0, -qd_slope],
                  [1.0, -qs_slope]], dtype=float)
    b = np.array([qd_intercept, qs_intercept], dtype=float)
    Q_eq, P_eq = linalg.solve(A, b)
    return float(Q_eq), float(P_eq)


# ---------- Forecasters ----------
class Forecaster(ABC):
    """Abstract forecaster: fit on a 1-D series, predict next `horizon` values."""

    @abstractmethod
    def fit(self, y: np.ndarray) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        ...

    def one_step_ahead_forecasts(self, y: np.ndarray, start: int) -> np.ndarray:
        """
        Rolling-origin one-step-ahead forecasts for indices start..len(y)-1.
        Uses only data up to each index to predict the next value.
        """
        preds = []
        for t in range(start, len(y)):
            self.fit(y[:t])
            preds.append(self.predict(1)[0])
        return np.asarray(preds)


class MovingAverageForecaster(Forecaster):
    """Simple k-period moving average, held constant for the horizon."""

    def __init__(self, window: int = 3):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._value = 0.0

    def fit(self, y: np.ndarray) -> "MovingAverageForecaster":
        y = np.asarray(y, dtype=float)
        if len(y) < self.window:
            raise ValueError(f"Need at least {self.window} observations")
        self._value = float(y[-self.window:].mean())
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._value)


class ExponentialSmoothingForecaster(Forecaster):
    """
    Simple exponential smoothing:
        s_0 = y_0
        s_t = α·y_t + (1 − α)·s_{t−1}
    Forecast = s_{last} for the whole horizon.
    """

    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._value = 0.0

    def fit(self, y: np.ndarray) -> "ExponentialSmoothingForecaster":
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("y is empty")
        s = y[0]
        for v in y[1:]:
            s = self.alpha * v + (1 - self.alpha) * s
        self._value = float(s)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._value)


class LinearTrendForecaster(Forecaster):
    """Fits y ~ slope·t + intercept, forecasts by extrapolating t."""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0
        self.n = 0

    def fit(self, y: np.ndarray) -> "LinearTrendForecaster":
        y = np.asarray(y, dtype=float)
        if len(y) < 2:
            raise ValueError("Need at least 2 observations")
        x = np.arange(len(y))
        self.slope, self.intercept = np.polyfit(x, y, 1)
        self.n = len(y)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        x = np.arange(self.n, self.n + horizon)
        return self.slope * x + self.intercept


class SeasonalNaiveForecaster(Forecaster):
    """
    Seasonal-naive: forecast for day t is the value at t − season_length.
    For day-11 forecast on a weekly pattern, use day 4 (11 − 7).
    """

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._last = np.array([])

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        y = np.asarray(y, dtype=float)
        if len(y) < self.season_length:
            raise ValueError("Need at least one full season of data")
        self._last = y[-self.season_length:].copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if len(self._last) == 0:
            raise RuntimeError("Call fit() first")
        reps = int(np.ceil(horizon / self.season_length))
        cycle = np.tile(self._last, reps)[:horizon]
        return cycle


# ---------- Metrics ----------
def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def grid_search_alpha(y: np.ndarray, start: int,
                      alphas: np.ndarray | None = None) -> tuple[float, float]:
    """
    Grid-search α for exponential smoothing using rolling one-step-ahead MAE.
    Returns (best_alpha, best_mae).
    """
    if alphas is None:
        alphas = np.arange(0.05, 1.00, 0.05)
    y = np.asarray(y, dtype=float)
    best_a, best_mae = 0.5, float("inf")
    for a in alphas:
        preds = ExponentialSmoothingForecaster(alpha=float(a)).one_step_ahead_forecasts(y, start)
        actual = y[start:]
        m = mae(actual, preds)
        if m < best_mae:
            best_mae, best_a = m, float(a)
    return best_a, best_mae


# ---------- Fleet sizing ----------
def fleet_size(forecast_passengers: float,
               trips_per_vehicle: int = 8,
               seats_per_trip: int = 14,
               buffer: float = 0.15) -> int:
    """Vehicles needed to carry forecast_passengers with a buffer, rounded up."""
    if forecast_passengers < 0:
        raise ValueError("passenger forecast cannot be negative")
    if trips_per_vehicle < 1 or seats_per_trip < 1:
        raise ValueError("trips_per_vehicle and seats_per_trip must be >= 1")
    if buffer < 0:
        raise ValueError("buffer must be >= 0")
    capacity_per_vehicle = trips_per_vehicle * seats_per_trip
    required = forecast_passengers * (1 + buffer) / capacity_per_vehicle
    return int(np.ceil(required))