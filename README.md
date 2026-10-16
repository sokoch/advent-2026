# MSCS & MSDS OOP with Python — Advent 2026 Mini-Projects

Five self-contained mini-projects implemented in Python using OOP, NumPy,
SciPy, the `statistics` module, and matplotlib. All logic lives in reusable
modules under `src/`, exercised by Jupyter notebooks under `notebooks/` and
unit tests under `tests/`.

## Setup

    python3 -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt

Requires Python 3.13 (tested on 3.13.3).

## Running

**Tests:**

    pytest tests/ -v

**Notebooks:**

    jupyter notebook notebooks/

Open any `projectN_*.ipynb` and use *Kernel → Restart & Run All*.

## Repository layout

    advent-2026/
    ├── src/                  Reusable logic (one module per project)
    │   ├── population.py     Mini-Project 1 — population forecaster
    │   ├── microgrid.py      Mini-Project 2 — solar micro-grid dispatch
    │   ├── fisheries.py      Mini-Project 3 — fish stock / price / risk
    │   ├── rainfall.py       Mini-Project 4 — region similarity & crops
    │   └── taxi.py           Mini-Project 5 — route revenue & fleet
    ├── tests/                pytest unit tests (62 total)
    ├── notebooks/            One notebook per project
    ├── data/                 Generated CSVs and figures (gitignored artefacts)
    ├── requirements.txt      Pinned dependencies
    └── README.md

## Findings summary

**1. UBOS District Population Forecaster.** Five districts (Kampala, Wakiso,
Gulu, Mbarara, Mbale) modelled with linear, CAGR, and Fibonacci forecasters.
Train 2015–2021, test 2022–2024. Linear trend wins on most districts; Fibonacci
grows unrealistically fast (ratios → φ ≈ 1.618). Forecasts for 2025–2029 imply
the largest classroom shortfall in Kampala and Wakiso.

**2. Solar Micro-Grid Dispatch.** The 2×2 system `[[3,2],[4,1]]` is well
conditioned (`cond ≈ 11.3`). 30 days of synthetic demand solved with
`scipy.linalg.solve`; the vectorised solve is several times faster than a
Python loop. Infeasible days are repaired with NNLS. Monte Carlo ±5 % demand
perturbation produces output spread consistent with `cond(A)`.

**3. Fisheries Stock, Price & Risk.** Logistic growth (`r = 0.4`, `K = 10,000`)
with proportional harvesting. `MSY = 1,000` tonnes/week. Harvest rates 0.05 and
0.10 leave the stock near equilibrium; 0.20 and 0.30 push it into a low-stock
regime. Revenue risk (CV-based) and 5 % VaR from 1,000 Monte Carlo paths are
reported in the notebook. Extension: closed season analysis.

**4. Rainfall Similarity & Crop Suitability.** Five regions, 12 monthly values
each. Cosine similarity implemented from scratch and verified against
`scipy.spatial.distance.cosine`, alongside Pearson correlation and Euclidean
distance. `scipy.signal.find_peaks` correctly classifies the south as bimodal
and the north as unimodal — matching Uganda's known climate zones. Extension:
real NASA POWER data with year-to-year box plots.

**5. Taxi Route Revenue, Pricing & Fleet Planner.** Three routes backtested with
moving average, exponential smoothing, and linear trend under rolling-origin
validation. Equilibrium analysis shows the 2,000 UGX Ntinda fare is below the
market-clearing 2,200 UGX, implying persistent excess demand. Day-11 revenue is
forecast per route and fleet sizes recommended with a 15 % buffer.
Seasonal-naive forecasting beats the moving average on simulated weekly-pattern
data.

## AI declaration

I used Claude (Anthropic), GitHub as coding assistants to help with boilerplate,
docstring wording, and debugging error messages. All logic was reviewed,
hand-verified against numerical examples and
discussed with the assistant before being accepted. No code was submitted
without understanding every line.

## Reproducibility

All random operations used `np.random.default_rng(seed)` with fixed seeds. Every
notebook runs top to bottom via *Restart & Run All* without errors. The test
suite covers normal cases and edge cases (empty input, negative values, singular
matrices, zero passengers, etc.).
