"""Central configuration for the curvature / tree-depth study."""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

# --- data source ---
# UCI Bike Sharing Dataset. Instead of reading a local copy of hour.csv, we
# download the official zip fresh from the source archive on every run and
# extract hour.csv (the hourly table) in memory. Nothing is read from a
# local file.
UCI_BIKESHARE_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
UCI_HOURLY_CSV_NAME = "hour.csv"


def load_bikeshare_hourly() -> pd.DataFrame:
    """Download the UCI Bike Sharing zip fresh from archive.ics.uci.edu and
    return the hourly table (hour.csv) as a DataFrame. Always pulls from the
    source -- no local file is read or cached."""
    with urlopen(UCI_BIKESHARE_ZIP_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        with zf.open(UCI_HOURLY_CSV_NAME) as f:
            return pd.read_csv(f)


# --- output location ---
# Save all study outputs (CSVs, npz, plots) to a folder under your own home
# directory -- Path.home() resolves automatically to whoever runs this
# script (e.g. C:/Users/<you> on Windows, /home/<you> or /Users/<you> on
# Linux/macOS), instead of a relative "artifacts" folder tied to wherever
# this script happens to live, or a path hardcoded for someone else's
# machine. Change the subfolder name below if you'd like results saved
# somewhere else.
OUT_DIR = Path.home() / "Downloads" / "bikeshare_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- target & features ---
TARGET = "cnt"
LOG1P_TARGET = True

# Dropped, with reason:
#   casual, registered -> components of cnt (target leakage)
#   instant, dteday    -> identifiers
#   atemp              -> ~0.99 correlated with temp
#   season             -> redundant with mnth
DROP = ["casual", "registered", "instant", "dteday", "atemp", "season"]

# Integer-coded categorical feature that is NOT cyclic/ordinal in any
# useful sense (1: clear ... 4: heavy rain -- there's a rough severity
# order, but no "wraparound" like hours or months have). Cast to pandas
# "category" dtype and modeled with XGBoost's native categorical split
# support (enable_categorical=True).
CAT_FEATURES = ["weathersit"]

# Integer-coded features that ARE cyclic: hour 23 is adjacent to hour 0,
# December is adjacent to January, Sunday is adjacent to Monday. Treating
# these as plain categories (or worse, as ordinary numeric splits) throws
# that adjacency away -- a depth-limited tree can only approximate
# "23 and 0 are neighbors" with several extra splits, if at all. Instead,
# each is replaced by a sin/cos pair: (sin(2*pi*x/period), cos(2*pi*x/period)).
# That maps the cycle onto a circle, so points near the wraparound end up
# close together in the 2D embedding the way they should be, and it's a
# genuinely continuous, differentiable encoding rather than an arbitrary
# ordering. period is the number of distinct values in one full cycle.
CYCLIC_FEATURES = {"mnth": 12, "hr": 24, "weekday": 7}


def add_cyclic_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each column named in CYCLIC_FEATURES with its sin/cos pair
    and drop the original integer-coded column. Returns a new DataFrame;
    does not mutate the input. Called identically from every script that
    prepares this data (the OOF screen and the fANOVA script) so the
    feature set -- and therefore feature_1/feature_2 names in the screened
    pairs CSV -- stays consistent between them."""
    df = df.copy()
    for col, period in CYCLIC_FEATURES.items():
        if col in df.columns:
            radians = 2 * np.pi * df[col].astype(float) / period
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)
            df = df.drop(columns=[col])
    return df


def feature_groups(columns) -> dict:
    """Map each LOGICAL feature name back to the list of physical column(s)
    that represent it, given the columns of a DataFrame that has already
    been through add_cyclic_encoding(). Cyclic features map to their
    [base_sin, base_cos] pair; every other column maps to a singleton list
    containing itself, e.g.:

        {"mnth": ["mnth_sin", "mnth_cos"], "weathersit": ["weathersit"], ...}

    The sin/cos split is an implementation detail of how a model consumes
    mnth/hr/weekday -- it shouldn't leak into anything OUTSIDE the actual
    fit/predict calls. Every other file (the screened-pairs CSV, the
    main-effect table, interaction-constraint groupings) should refer to
    features by their logical name ("hr"), and use this mapping only at
    the point a model is being built or evaluated, to expand that name
    back into whichever physical column(s) currently represent it.
    """
    groups = {}
    for base in CYCLIC_FEATURES:
        sin_c, cos_c = f"{base}_sin", f"{base}_cos"
        if sin_c in columns and cos_c in columns:
            groups[base] = [sin_c, cos_c]
    handled = {c for pair in groups.values() for c in pair}
    for c in columns:
        if c not in handled:
            groups[c] = [c]
    return groups

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
SWEEP_JOBS = None              # concurrent CatBoost fits (None = auto: ~cores)
