"""Unit tests for src/taxi.py — Mini-Project 5."""
import numpy as np
import pytest

from src.taxi import (
    Route, solve_equilibrium,
    MovingAverageForecaster, ExponentialSmoothingForecaster,
    LinearTrendForecaster, SeasonalNaiveForecaster,
    grid_search_alpha, fleet_size, mae,
)


# ---------- Route ----------

def test_route_revenue_and_stats():
    r = Route("Test", [10, 20, 30], fare=1000)
    assert np.allclose(r.daily_revenue(), [10_000, 20_000, 30_000])
    assert r.total_revenue() == pytest.approx(60_000)
    s = r.stats()
    assert s["mean"] == pytest.approx(20_000)


def test_route_rejects_negative_passengers():
    with pytest.raises(ValueError):
        Route("Bad", [10, -1, 20], fare=1000)


def test_route_rejects_empty():
    with pytest.raises(ValueError):
        Route("Empty", [], fare=1000)


# ---------- Equilibrium ----------

def test_equilibrium_solution():
    # Qd = 120 - 0.02 P ; Qs = 10 + 0.03 P
    # At equilibrium: 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P = 2200
    # Q = 120 - 0.02*2200 = 76
    Q, P = solve_equilibrium()
    assert P == pytest.approx(2200.0)
    assert Q == pytest.approx(76.0)


# ---------- Forecasters ----------

def test_moving_average_of_constant_series():
    f = MovingAverageForecaster(window=3).fit(np.array([5, 5, 5, 5]))
    assert np.allclose(f.predict(3), [5, 5, 5])


def test_moving_average_of_increasing_series():
    f = MovingAverageForecaster(window=3).fit(np.array([1, 2, 3, 4, 5]))
    # last 3 = 3,4,5 -> mean = 4
    assert np.allclose(f.predict(2), [4, 4])


def test_exponential_smoothing_alpha_one_returns_last():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    f = ExponentialSmoothingForecaster(alpha=1.0).fit(y)
    assert f.predict(1)[0] == pytest.approx(5.0)


def test_exponential_smoothing_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=0.0)
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=1.5)


def test_linear_trend_on_line():
    y = np.array([2, 4, 6, 8], dtype=float)
    f = LinearTrendForecaster().fit(y)
    # next two values continue: 10, 12
    assert np.allclose(f.predict(2), [10, 12], atol=1e-9)


def test_seasonal_naive_repeats_season():
    y = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)  # one weekly season
    f = SeasonalNaiveForecaster(season_length=7).fit(y)
    # Next 7 = the same as the last 7
    assert np.allclose(f.predict(7), [1, 2, 3, 4, 5, 6, 7])


# ---------- Grid search ----------

def test_grid_search_returns_valid_alpha():
    rng = np.random.default_rng(0)
    y = 100 + rng.normal(0, 3, 20)
    alpha, m = grid_search_alpha(y, start=5)
    assert 0 < alpha <= 1
    assert m >= 0


# ---------- Fleet sizing ----------

def test_fleet_size_basic():
    # 448 passengers, 8*14 = 112 per vehicle -> 448/112 = 4
    # With 15% buffer: 448 * 1.15 / 112 = 4.6 -> ceil = 5
    assert fleet_size(448, trips_per_vehicle=8, seats_per_trip=14, buffer=0.15) == 5


def test_fleet_size_zero_passengers():
    assert fleet_size(0, trips_per_vehicle=8, seats_per_trip=14, buffer=0.15) == 0


def test_fleet_size_rejects_negative():
    with pytest.raises(ValueError):
        fleet_size(-1)


# ---------- MAE ----------

def test_mae_basic():
    assert mae(np.array([1, 2, 3]), np.array([1, 3, 2])) == pytest.approx(2/3)