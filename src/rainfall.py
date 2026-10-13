"""
Rainfall similarity and crop suitability.
Mini-Project 4: Region statistics, crop rules, similarity metrics, season detection.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.distance import cosine as sp_cosine
from scipy.signal import find_peaks


class Region:
    """
    A region with 12 monthly rainfall values (mm).

    Attributes
    ----------
    name : str
    rainfall : np.ndarray, shape (12,)
        Monthly rainfall in mm, January..December.
    """

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def __init__(self, name: str, rainfall: Iterable[float]):
        self.name = name
        self.rainfall = np.asarray(rainfall, dtype=float)
        if self.rainfall.ndim != 1 or len(self.rainfall) != 12:
            raise ValueError("rainfall must have exactly 12 monthly values")
        if np.any(self.rainfall < 0):
            raise ValueError("rainfall cannot be negative")

    def __repr__(self) -> str:
        return f"Region(name={self.name!r}, annual_total={self.annual_total():.1f} mm)"

    def __len__(self) -> int:
        return 12

    # ---------- Statistics ----------
    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean_monthly(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return self.MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return self.MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation = std / mean (dimensionless)."""
        m = self.rainfall.mean()
        if m <= 0:
            return float("nan")
        # ddof=1 for sample standard deviation
        return float(self.rainfall.std(ddof=1) / m)

    # ---------- Season detection ----------
    def detect_peaks(self, prominence: float = 20.0) -> list[int]:
        """Return 1-based month indices of rainy peaks."""
        peaks, _ = find_peaks(self.rainfall, prominence=prominence)
        return (peaks + 1).tolist()

    def season_type(self, prominence: float = 20.0) -> str:
        n = len(self.detect_peaks(prominence=prominence))
        if n == 0:
            return "no clear peak"
        if n == 1:
            return "unimodal"
        return "bimodal"


class CropRule:
    """
    Suitable monthly rainfall range for a crop (mm/month).

    Thresholds are adapted from FAO Irrigation & Drainage Paper 56
    (Allen et al., 1998) monthly crop water requirements, rounded to
    convenient round numbers.
    """

    def __init__(self, crop: str, low: float, high: float):
        if low < 0 or high <= low:
            raise ValueError("Require 0 <= low < high")
        self.crop = crop
        self.low = float(low)
        self.high = float(high)

    def classify(self, mm: float) -> str:
        if mm < self.low:
            return "Drought risk"
        if mm > self.high:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def classify_series(self, monthly: np.ndarray) -> list[str]:
        return [self.classify(v) for v in monthly]


# ---------- Similarity metrics ----------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Proper cosine similarity between two vectors (NOT math.cos).

    cos_sim(a, b) = (a . b) / (||a|| * ||b||)
    Range: [-1, 1]; for non-negative rainfall it lies in [0, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float)
                                - np.asarray(b, dtype=float)))


def verify_cosine_against_scipy(a: np.ndarray, b: np.ndarray) -> bool:
    """
    scipy.spatial.distance.cosine returns the COSINE DISTANCE = 1 - cos_sim.
    Verify our cosine_similarity matches scipy's distance-based form.
    """
    our_sim = cosine_similarity(a, b)
    scipy_dist = float(sp_cosine(a, b))
    return abs((1.0 - scipy_dist) - our_sim) < 1e-9