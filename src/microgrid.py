"""
Module for the solar micro-grid dispatch planner.
Mini-Project 2: Solving the 2x2 linear system per day.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import linalg
from scipy.optimize import nnls


class MicroGrid:
    """
    Two-source micro-grid (solar + battery).

    Solves  A @ [x, y] = [D1, D2]  per day, where:
      x = energy drawn from solar (kWh)
      y = energy drawn from battery (kWh)
      D1 = daytime load (kWh)
      D2 = critical-equipment load (kWh)

    Parameters
    ----------
    A : array-like, shape (2, 2)
        Coefficient matrix.
    solar_cost : float
        Cost per kWh from solar (UGX).
    battery_cost : float
        Cost per kWh from battery (UGX).
    """

    def __init__(self, A: Iterable, solar_cost: float = 150.0, battery_cost: float = 450.0):
        self.A = np.asarray(A, dtype=float)
        if self.A.shape != (2, 2):
            raise ValueError("A must be 2x2")
        self.solar_cost = float(solar_cost)
        self.battery_cost = float(battery_cost)

    # ---------- Diagnostics ----------
    @property
    def determinant(self) -> float:
        return float(np.linalg.det(self.A))

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.A))

    def is_well_posed(self) -> bool:
        return abs(self.determinant) > 1e-12

    # ---------- Solver ----------
    def solve_day(self, d1: float, d2: float) -> np.ndarray:
        """Return [x, y] for a single day's demands."""
        if d1 < 0 or d2 < 0:
            raise ValueError("Demands must be non-negative")
        if not self.is_well_posed():
            raise ValueError("System is singular; cannot solve")
        return linalg.solve(self.A, np.array([d1, d2], dtype=float))

    def solve_many_loop(self, D: np.ndarray) -> np.ndarray:
        """D is (2, n). Returns (2, n) via a Python loop."""
        n = D.shape[1]
        out = np.empty((2, n))
        for i in range(n):
            out[:, i] = linalg.solve(self.A, D[:, i])
        return out

    def solve_many_vectorised(self, D: np.ndarray) -> np.ndarray:
        """D is (2, n). Single linalg.solve call."""
        return linalg.solve(self.A, D)

    def solve_nnls(self, d1: float, d2: float) -> np.ndarray:
        """Non-negative least squares — returns [x, y] with x, y >= 0."""
        b = np.array([d1, d2], dtype=float)
        sol, _ = nnls(self.A, b)
        return sol

    # ---------- Cost ----------
    def daily_cost(self, sol: np.ndarray) -> float:
        """Given [x, y] in kWh, return UGX cost."""
        return float(sol[0] * self.solar_cost + sol[1] * self.battery_cost)


class HybridMicroGrid(MicroGrid):
    """
    Three-source micro-grid (solar + battery + diesel).

    Solves A3 @ [x, y, z] = [D1, D2, D3].
    """

    def __init__(self, A: Iterable, diesel_cost: float = 1200.0,
                 solar_cost: float = 150.0, battery_cost: float = 450.0):
        super().__init__(np.eye(2), solar_cost=solar_cost, battery_cost=battery_cost)
        self.A = np.asarray(A, dtype=float)
        if self.A.shape != (3, 3):
            raise ValueError("Hybrid A must be 3x3")
        self.diesel_cost = float(diesel_cost)

    def solve_day(self, d1: float, d2: float, d3: float) -> np.ndarray:  # type: ignore[override]
        if min(d1, d2, d3) < 0:
            raise ValueError("Demands must be non-negative")
        return linalg.solve(self.A, np.array([d1, d2, d3], dtype=float))

    def daily_cost(self, sol: np.ndarray) -> float:  # type: ignore[override]
        return float(sol[0] * self.solar_cost
                     + sol[1] * self.battery_cost
                     + sol[2] * self.diesel_cost)