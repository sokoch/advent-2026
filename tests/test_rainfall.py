"""Unit tests for src/rainfall.py — Mini-Project 4."""
import numpy as np
import pytest

from src.rainfall import (
    Region, CropRule,
    cosine_similarity, pearson_correlation, euclidean_distance,
    verify_cosine_against_scipy,
)


RAIN_A = np.array([50, 60, 100, 150, 120, 60,  40,  45,  80, 130, 110, 70])
RAIN_B = np.array([40, 55,  95, 160, 130, 70,  45,  50,  85, 140, 120, 75])


# ---------- Region ----------

def test_region_basic_statistics():
    r = Region("Test", RAIN_A)
    assert r.annual_total() == pytest.approx(RAIN_A.sum())
    assert r.mean_monthly() == pytest.approx(RAIN_A.mean())


def test_region_wettest_driest():
    r = Region("Test", RAIN_A)
    # argmax is index 3 -> April
    assert r.wettest_month() == "Apr"
    # argmin is index 6 -> July
    assert r.driest_month() == "Jul"


def test_region_rejects_wrong_length():
    with pytest.raises(ValueError):
        Region("Bad", [1, 2, 3])


def test_region_rejects_negative():
    bad = RAIN_A.copy()
    bad[0] = -5
    with pytest.raises(ValueError):
        Region("Bad", bad)


# ---------- CropRule ----------

def test_crop_rule_classification():
    maize = CropRule("maize", 50, 150)
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(20)  == "Drought risk"
    assert maize.classify(200) == "Waterlogging risk"


def test_crop_rule_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        CropRule("maize", 100, 50)


# ---------- Similarity metrics ----------

def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity(RAIN_A, RAIN_A) == pytest.approx(1.0)


def test_cosine_similarity_matches_scipy_distance():
    assert verify_cosine_against_scipy(RAIN_A, RAIN_B)


def test_pearson_correlation_of_identical_vectors_is_one():
    assert pearson_correlation(RAIN_A, RAIN_A) == pytest.approx(1.0)


def test_euclidean_distance_zero_for_identical():
    assert euclidean_distance(RAIN_A, RAIN_A) == 0


# ---------- Season detection ----------

def test_season_detection_unimodal():
    # Single clear peak in May, monotone tails on either side
    rain = np.array([10, 15, 40, 120, 200, 60, 30, 20, 15, 12, 10, 8])
    assert Region("U", rain).season_type(prominence=20) == "unimodal"

def test_season_detection_bimodal():
    rain = np.array([30, 40, 120, 100, 50, 30, 20, 60, 110, 80, 40, 30])
    assert Region("B", rain).season_type(prominence=20) == "bimodal"