"""Unit tests for src/microgrid.py — Mini-Project 2."""
import numpy as np
import pytest

from src.microgrid import MicroGrid, HybridMicroGrid


A2 = np.array([[3, 2], [4, 1]], dtype=float)
A3 = np.array([[3, 2, 1], [4, 1, 1], [1, 1, 1]], dtype=float)


# ---------- Normal cases ----------

def test_determinant_and_condition():
    g = MicroGrid(A2)
    assert g.determinant == pytest.approx(3*1 - 2*4)   # -5
    assert g.condition_number > 1
    assert g.is_well_posed()


def test_solve_known_system():
    # 3x + 2y = 8 ; 4x + y = 10  -> x = 12/5 = 2.4, y = 2/5 = 0.4
    g = MicroGrid(A2)
    sol = g.solve_day(8, 10)
    assert np.allclose(sol, [2.4, 0.4])


def test_cost_computation():
    g = MicroGrid(A2, solar_cost=100.0, battery_cost=500.0)
    cost = g.daily_cost(np.array([2.0, 3.0]))
    assert cost == pytest.approx(2*100 + 3*500)


def test_loop_and_vectorised_match():
    g = MicroGrid(A2)
    rng = np.random.default_rng(0)
    D = rng.uniform(10, 100, size=(2, 30))
    a = g.solve_many_loop(D)
    b = g.solve_many_vectorised(D)
    assert np.allclose(a, b)


# ---------- Edge cases ----------

def test_rejects_negative_demand():
    g = MicroGrid(A2)
    with pytest.raises(ValueError):
        g.solve_day(-1, 5)


def test_singular_matrix_rejected():
    g = MicroGrid(np.array([[1, 2], [2, 4]], dtype=float))
    assert not g.is_well_posed()
    with pytest.raises(ValueError):
        g.solve_day(1, 1)


def test_nnls_returns_non_negative():
    g = MicroGrid(A2)
    # Pick demands that force a negative component under plain solve
    sol = g.solve_nnls(1, 20)
    assert (sol >= 0).all()


# ---------- Hybrid subclass ----------

def test_hybrid_solves_3x3():
    h = HybridMicroGrid(A3)
    sol = h.solve_day(10, 10, 5)
    assert len(sol) == 3
    # Verify A3 @ sol = b
    assert np.allclose(A3 @ sol, [10, 10, 5])


def test_hybrid_cost_includes_diesel():
    h = HybridMicroGrid(A3, solar_cost=100, battery_cost=500, diesel_cost=1000)
    sol = np.array([1.0, 2.0, 3.0])
    assert h.daily_cost(sol) == pytest.approx(1*100 + 2*500 + 3*1000)