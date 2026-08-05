"""Central configuration for the California Housing curvature / tree-depth
study.

Counterpart to bikeshare_config.py, kept completely separate: its own
output folder, its own data loader, no shared state with the bike-share
study. Nothing in this file touches bikeshare_config.py, the
bikeshare_artifacts folder, or any bikeshare_hourly_* file.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

# --- data source ---
# sklearn's California Housing dataset is fetched via
# sklearn.datasets.fetch_california_housing(), which downloads and caches
# the data itself (in scikit-learn's own cache directory) the first time
# it's called -- no separate zip/URL handling needed here, unlike the
# bike-share study's manual UCI fetch.
def load_california_housing():
    """Return (X, y_raw): the raw feature DataFrame and the raw
    (untransformed) median house value target as a 1-D numpy array."""
    data = fetch_california_housing(as_frame=True)
    X = data.data.copy()
    y_raw = data.target.to_numpy(float)
    return X, y_raw


# --- output location ---
# A DIFFERENT folder from the bike-share study's OUT_DIR
# (Downloads/bikeshare_artifacts), so neither study's screened pairs or
# results can ever overwrite the other's.
OUT_DIR = Path.home() / "Downloads" / "california_housing_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- target ---
TARGET = "MedHouseVal"
LOG1P_TARGET = True   # right-skewed home values; log1p is the usual
                       # variance-stabilizing choice (same treatment as
                       # the bike-share study's cnt target)

# No columns to drop: fetch_california_housing's 8 features (MedInc,
# HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
# Longitude) contain no leakage components, identifiers, or redundant
# pairs the way the bike-share raw table does. Kept as an empty list
# (rather than removed) so both scripts' preprocessing code stays
# structurally identical to the bike-share versions -- easy to add an
# entry here later if that ever changes.
DROP = []

# No categorical features: every column in this dataset is a continuous
# numeric quantity (income, age, room counts, population, coordinates).
CAT_FEATURES = []

# No cyclic features: nothing here wraps around the way hour-of-day or
# month-of-year does, so unlike bikeshare_config.py there is no
# CYCLIC_FEATURES dict, no add_cyclic_encoding(), and no feature_groups()
# helper in this file -- every feature is already its own single physical
# column, and the two consumer scripts treat it that way directly.

# --- reference model that defines the geometry (kernel + curvature) ---
REF_DEPTH = 2
REF_ITERS = 500
REF_LR = 0.05

# --- Nystrom kernel approximation ---
N_LANDMARKS = 500
LANDMARK_SPACE = "embedding"   # k-means on the leaf-indicator space Phi

# --- graph / Ollivier-Ricci ---
KNN = 15
OR_ALPHA = 0.0                 # idleness 0: measure is all on neighbors, allows
                               # negative curvature (boundary/high-complexity regions)

# --- depth sweep for "required depth" (independent axis) ---
SWEEP_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # extended so best depth isn't censored
ELBOW_FRAC = 0.90              # required depth = shallowest reaching this frac of gain
SWEEP_ITERS = 500
SWEEP_LR = 0.05
STAB_TOL_FRAC = 0.05           # tol = frac * std(y_log): prediction "settled"
SWEEP_CV = 5                   # cross-fitted OOF predictions (None = in-sample)
SWEEP_JOBS = None              # concurrent fits (None = auto: ~cores)
