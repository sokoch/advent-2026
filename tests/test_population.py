"""Unit tests for src/population.py — Mini-Project 1."""
import numpy as np
import pytest

from src.population import (
    DistrictPopulation,
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciForecaster,
    mae,
    rmse,
    mape,
)


# ---------- DistrictPopulation: normal cases ----------

def test_district_population_basic():
    d = DistrictPopulation("Test", [2015, 2016, 2017], [100, 110, 121])
    assert len(d) == 3
    assert "Test" in repr(d)


def test_cagr_matches_hand_calculation():
    # 100 -> 121 over 2 years => CAGR = (121/100)^(1/2) - 1 = 0.10
    d = DistrictPopulation("X", [2015, 2016, 2017], [100, 110, 121])
    assert abs(d.cagr() - 0.10) < 1e-9


def test_growth_rates_length_and_values():
    d = DistrictPopulation("X", [2015, 2016, 2017], [100, 110, 121])
    rates = d.growth_rates()
    assert len(rates) == 2
    assert rates[0] == pytest.approx(0.10)
    assert rates[1] == pytest.approx(0.10)


# ---------- DistrictPopulation: edge cases ----------

def test_district_population_rejects_negative():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2015, 2016], [100, -5])


def test_district_population_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2015, 2016, 2017], [100, 110])


def test_district_population_rejects_empty():
    with pytest.raises(ValueError):
        DistrictPopulation("Empty", [], [])


def test_district_population_rejects_non_increasing_years():
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", [2015, 2015, 2017], [100, 110, 121])


# ---------- Forecasters ----------

def test_linear_forecaster_predicts_on_trend():
    years = np.array([2015, 2016, 2017, 2018])
    values = np.array([10, 20, 30, 40])
    f = LinearTrendForecaster().fit(years, values)
    preds = f.predict(2)
    assert np.allclose(preds, [50, 60], atol=1e-6)


def test_cagr_forecaster_predicts_compound():
    years = np.array([2015, 2016, 2017])
    values = np.array([100, 110, 121])
    f = CAGRForecaster().fit(years, values)
    preds = f.predict(2)
    assert np.allclose(preds, [133.1, 146.41], atol=1e-6)


def test_fibonacci_forecaster_first_step():
    years = np.array([2015, 2016])
    values = np.array([100, 100])
    f = FibonacciForecaster().fit(years, values)
    preds = f.predict(5)
    # First ratio F(3)/F(2) = 2/1 = 2 -> 200
    assert preds[0] == pytest.approx(200.0)


def test_forecaster_predict_before_fit_raises():
    f = LinearTrendForecaster()
    with pytest.raises(RuntimeError):
        f.predict(3)


# ---------- Metrics ----------

def test_metrics_zero_error():
    a = np.array([1, 2, 3])
    assert mae(a, a) == 0
    assert rmse(a, a) == 0
    assert mape(a, a) == 0


def test_mape_handles_zero_actual():
    actual = np.array([0, 10, 20])
    predicted = np.array([1, 12, 22])
    result = mape(actual, predicted)
    expected = np.mean([abs(10 - 12) / 10, abs(20 - 22) / 20]) * 100
    assert result == pytest.approx(expected)