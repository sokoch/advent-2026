"""Unit tests for src/fisheries.py — Mini-Project 3."""
import numpy as np
import pytest

from src.fisheries import FishStock, PriceModel, RiskAssessor


# ---------- FishStock ----------

def test_fish_stock_simulation_length():
    fs = FishStock()
    N = fs.simulate(weeks=52)
    assert len(N) == 53
    assert N[0] == 4_000.0


def test_fish_stock_msy_formula():
    fs = FishStock(r=0.4, K=10_000)
    # MSY = rK/4 = 1000
    assert fs.msy() == pytest.approx(1_000.0)


def test_fish_stock_rejects_n0_above_K():
    with pytest.raises(ValueError):
        FishStock(r=0.4, K=1000, N0=2000, h=0.1)


def test_fish_stock_rejects_negative_h():
    with pytest.raises(ValueError):
        FishStock(h=-0.1)


def test_fish_stock_rejects_zero_weeks():
    fs = FishStock()
    with pytest.raises(ValueError):
        fs.simulate(weeks=0)


# ---------- PriceModel ----------

def test_price_model_length_and_bounds():
    pm = PriceModel(seed=0)
    p = pm.simulate(weeks=52)
    assert len(p) == 52
    assert (p >= 9_000).all() and (p <= 16_000).all()


def test_price_model_first_value_equals_start():
    pm = PriceModel(start=12_000, seed=0)
    p = pm.simulate(weeks=52)
    assert p[0] == 12_000


def test_price_model_rejects_bad_bounds():
    with pytest.raises(ValueError):
        PriceModel(low=10_000, high=9_000)


def test_price_model_paths_shape():
    pm = PriceModel(seed=1)
    paths = pm.simulate_paths(weeks=10, n_paths=5)
    assert paths.shape == (5, 10)


# ---------- RiskAssessor ----------

def test_risk_assessor_classifies_low():
    ra = RiskAssessor(low_cv=0.10, high_cv=0.25)
    revenues = np.array([100, 101, 99, 100, 102, 98])   # very low CV
    assert ra.classify(revenues) == "Low"


def test_risk_assessor_classifies_high():
    ra = RiskAssessor(low_cv=0.10, high_cv=0.25)
    revenues = np.array([10, 100, 50, 200, 30, 150])    # high CV
    assert ra.classify(revenues) == "High"


def test_value_at_risk():
    ra = RiskAssessor()
    revenues = np.arange(1, 101, dtype=float)  # 1..100
    var5 = ra.value_at_risk(revenues, alpha=0.05)
    assert var5 == pytest.approx(np.percentile(revenues, 5))


def test_risk_assessor_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        RiskAssessor(low_cv=0.30, high_cv=0.20)