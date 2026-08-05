"""Central configuration for the synthetic curvature/importance validation
study.

Purpose: this dataset is DESIGNED, not observed. 8 features, 4 disjoint
pairs, each pair driving a different, deliberately-shaped interaction
component of the target -- with curvature (shape narrowness) and variance
(amplitude) controlled INDEPENDENTLY so the four combinations of
high/low curvature x high/low variance are all represented:

    pair        pattern        curvature   variance   importance
    (f1, f2)    small bump     high        low        low
    (f3, f4)    smooth shift   low         high       high
    (f5, f6)    tall peak      high        high       highest
    (f7, f8)    blip           low         low        unimportant (lowest)

The point is to run the SAME screen -> fANOVA kernel-curvature pipeline
used for the bike-share and California-housing studies on data where the
ground truth is known by construction, as a check on whether the
pipeline's model_kernel_curvature() (Q_S) and anova_energy actually
recover this intended structure.

Kept completely separate from the other two studies: its own output
folder, its own data generator (no external download), no shared state.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# --- output location ---
OUT_DIR = Path.home() / "Downloads" / "synthetic_curvature_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- target ---
TARGET = "y"
LOG1P_TARGET = False   # the target can be negative (tanh/Gaussian components
                        # centered near 0 plus noise), so log1p doesn't apply
                        # here the way it does for the strictly-positive
                        # housing-price / ride-count targets in the other
                        # two studies.

DROP = []
CAT_FEATURES = []
# No cyclic features -- this is synthetic, continuous-only data.

# --- synthetic data generation ---
N_SAMPLES = 20_000
FEATURE_RANGE = (-3.0, 3.0)   # each of f1..f8 ~ Uniform(FEATURE_RANGE)
NOISE_STD = 0.10

# Shape parameters: sigma controls the two narrow (high-curvature)
# components' width; s_broad controls the two broad (low-curvature)
# components' steepness. Both pairs sharing a shape parameter is
# deliberate -- it isolates curvature as a shape-only property, decoupled
# from amplitude, matching model_kernel_curvature()'s documented
# amplitude-invariance.
SIGMA_NARROW = 0.4     # width of the small-bump / tall-peak Gaussians
S_BROAD = 3.0           # steepness scale of the smooth-shift / blip tanh ramps

# Amplitudes: control variance independently of the shape parameters above.
A_BUMP = 0.3     # small bump: small amplitude -> low variance
A_SHIFT = 1.5    # smooth shift: large amplitude -> high variance
A_PEAK = 6.0     # tall peak: large amplitude, same narrow shape as bump -> high variance
A_BLIP = 0.05    # blip: tiny amplitude -> lowest variance, "unimportant"

# Ground truth: which pair drives which pattern, and what we expect to see.
# Used by the fANOVA script to annotate its output -- purely diagnostic,
# not fed into modeling in any way (the pipeline sees only f1..f8 and y).
GROUND_TRUTH = {
    ("f1", "f2"): {"pattern": "small_bump",   "expected_curvature": "high", "expected_importance": "low"},
    ("f3", "f4"): {"pattern": "smooth_shift", "expected_curvature": "low",  "expected_importance": "high"},
    ("f5", "f6"): {"pattern": "tall_peak",    "expected_curvature": "high", "expected_importance": "highest"},
    ("f7", "f8"): {"pattern": "blip",         "expected_curvature": "low",  "expected_importance": "unimportant (lowest)"},
}

# --- model hyperparameters (same roles as in the bike-share / California
# Housing configs) ---
# GAM stage (depth 10, additive) and deep residual stage (depth 6,
# constrained to the known pairs) in the fANOVA script:
SWEEP_ITERS = 500
SWEEP_LR = 0.05

# Depth-2 reference/screening model (only needed if you run the OOF screen
# script; unused when going straight to the fANOVA script with the known
# pairs):
REF_DEPTH = 2
REF_ITERS = 500
REF_LR = 0.05
SWEEP_CV = 5

# kNN graph size for model_kernel_curvature()'s roughness/curvature
# computation:
KNN = 15


def generate_synthetic_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE):
    """Generate the synthetic 8-feature dataset. Returns (X, y) where X is
    a DataFrame with columns f1..f8 and y is a 1-D numpy array.

    y = bump(f1,f2) + shift(f3,f4) + peak(f5,f6) + blip(f7,f8) + noise

    Each term is a genuine (non-additively-separable) interaction: a
    nonlinear function of a combination of its two inputs, so none of it
    can be explained away by main effects of f1..f8 alone -- the fANOVA
    interaction component A_S for each of these 4 pairs should be where
    essentially all of the corresponding signal shows up.
    """
    rng = np.random.default_rng(seed)
    lo, hi = FEATURE_RANGE
    cols = {f"f{i}": rng.uniform(lo, hi, size=n) for i in range(1, 9)}
    X = pd.DataFrame(cols)

    # SMALL BUMP (f1, f2): narrow 2D Gaussian, small amplitude.
    # High curvature (narrow => large local second derivative where it's
    # non-negligible), low variance (small amplitude, and the bump covers
    # only a small fraction of the domain).
    bump = A_BUMP * np.exp(-(X["f1"] ** 2 + X["f2"] ** 2) / (2 * SIGMA_NARROW ** 2))

    # SMOOTH SHIFT (f3, f4): broad tanh ramp of the SUM f3+f4. Nonlinear in
    # a linear combination of both inputs => genuine interaction (an
    # affine function of f3+f4 would be purely additive/no interaction;
    # tanh is not affine, so this has a real fANOVA interaction component).
    # Low curvature (broad, gentle slope across the domain), high variance
    # (large amplitude, and the ramp sweeps its full range across most of
    # the domain rather than being localized).
    shift = A_SHIFT * np.tanh((X["f3"] + X["f4"]) / S_BROAD)

    # TALL PEAK (f5, f6): SAME narrow shape as the bump (same SIGMA_NARROW
    # => same curvature profile), but a much larger amplitude. High
    # curvature AND high variance -> should be the most important
    # component overall.
    peak = A_PEAK * np.exp(-(X["f5"] ** 2 + X["f6"] ** 2) / (2 * SIGMA_NARROW ** 2))

    # BLIP (f7, f8): SAME broad shape as the smooth shift (same S_BROAD,
    # using the DIFFERENCE f7-f8 instead of the sum just so it isn't a
    # literal duplicate of the shift term), but a tiny amplitude. Low
    # curvature AND low variance -> should be the least important
    # component, more so than the small bump.
    blip = A_BLIP * np.tanh((X["f7"] - X["f8"]) / S_BROAD)

    noise = rng.normal(0.0, NOISE_STD, size=n)

    y = (bump + shift + peak + blip).to_numpy() + noise
    return X, y
