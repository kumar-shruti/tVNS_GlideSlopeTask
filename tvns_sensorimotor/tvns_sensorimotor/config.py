"""
Central configuration: file paths, experiment constants, plotting colours, and
the knobs referenced by the reviewer checklist (Quick_Checklist_by_Action.md).

Nothing in this file reads data. Edit `DATA` paths to point at your copies, or
set the environment variable ``TVNS_DATA_DIR`` before importing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
_PERFORMANCE_CSV = "SDCP_flight-performance-metrics_2023-11-02.xlsx - Cohort2 (1).csv"
# Default: the first of (project folder above the package, package root, the
# package itself) that holds the performance CSV; falls back to the first.
_HERE = Path(__file__).resolve()
_CANDIDATE_DATA_DIRS = (_HERE.parents[2], _HERE.parents[1], _HERE.parent)
_DEFAULT_DATA_DIR = next((d for d in _CANDIDATE_DATA_DIRS if (d / _PERFORMANCE_CSV).exists()),
                         _CANDIDATE_DATA_DIRS[0])
DATA_DIR = Path(os.environ.get("TVNS_DATA_DIR", _DEFAULT_DATA_DIR))
OUT_DIR = Path(os.environ.get("TVNS_OUT_DIR", _HERE.parents[1] / "outputs"))


@dataclass(frozen=True)
class DataPaths:
    """Resolved input files. Missing physio/EEG-feature tables are tolerated;
    the modules that need them raise a clear error only when actually called."""

    performance: Path = DATA_DIR / _PERFORMANCE_CSV
    # physio path in pkl
    physio: Path = DATA_DIR / "nn_data.pkl"

    def require(self, name: str) -> Path:
        p = getattr(self, name)
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Required input '{name}' not found at {p}. "
                f"Set tvns_sensorimotor.config.DATA.{name} or TVNS_DATA_DIR."
            )
        return Path(p)


DATA = DataPaths()

# --------------------------------------------------------------------------- #
# Experiment structure                                                         #
# --------------------------------------------------------------------------- #
BASELINE_DAY = 0            # D00
RUNS_PER_BLOCK = 6          # 6 runs -> 1 block
BLOCKS_PER_DAY = 2          # runs 1–6 and 7–12
N_TRAINING_BLOCKS = 8       # 4 training days x 2
FOLLOWUP_DAY_PREFIX = "DR"  # DR1 / DR2 retention days are excluded from the main analyses

# Composite "needle" error = Euclidean norm of glideslope + localizer deviation
# (W4: this is an instrument-deviation measure, not a joystick measure).
NEEDLE_COMPONENTS = ("RMSE_GSNeedle", "RMSE_LOCNeedle")

BEHAVIOR_MEASURES = ("RMSE_Needle", "RMSE_Airspeed")
PHYSIO_MEASURES = ("Heart Rate", "RMSSD", "Pupil Diameter")

# --------------------------------------------------------------------------- #
# Sign convention  (checklist B1)                                              #
# --------------------------------------------------------------------------- #
# improvement = baseline_mean - run_value   ==>  POSITIVE = better than baseline.
# prep.add_baseline_correction() asserts Figure-2A positivity against this.
IMPROVEMENT_IS_BASELINE_MINUS_RUN = True

# --------------------------------------------------------------------------- #
# Statistics knobs (checklist S1, B6, B10, S4)                                 #
# --------------------------------------------------------------------------- #
ALPHA = 0.05
MULTIPLICITY_METHOD = "bonferroni"   # applied across the 8 blocks, per coefficient
REML = True
LMM_OPTIMIZER = "powell"
LMM_MAXITER = 500

# B6: Table 2 physiological models. Manuscript Results + Fig 3 caption say
# "random intercepts and random slopes"; eq. (3) says intercept only. We fit
# random intercept + random slope on centred trial and report that consistently.
PHYSIO_RE_FORMULA = "~ trial_c"

# B10: within-block subject slopes for the Stim-vs-Sham comparison must come from
# an *independent* OLS fit per subject with NO treatment term (not BLUPs from a
# model that already contains treatment). Toggle only to reproduce the old path.
SLOPE_METHOD = "independent_ols"     # {"independent_ols", "blup_legacy"}
MIN_TRIALS_FOR_SLOPE = 4             # runs per block needed to estimate a slope

# S4: RMSSD is right-skewed and was truncated at >200 ms before fitting.
RMSSD_LOG_MODEL = True               # additionally fit log(RMSSD) ~ ...
RMSSD_TRUNCATION_MS = (5.0, 200.0)   # legacy pre-fit truncation window
RMSSD_SENSITIVITY_NO_TRUNCATION = True

HR_PLAUSIBLE_BPM = (40.0, 200.0)
PUPIL_PLAUSIBLE_MM = (1.5, 9.0)

# --------------------------------------------------------------------------- #
# EEG knobs (checklist B5, B7, M1)                                             #
# --------------------------------------------------------------------------- #
EEG_FS = 150
EEG_N_CHANNELS = 64
EEG_MONTAGE = "biosemi64"

# B5: continuous EEG was band-passed 1–60 Hz with 60/120 Hz notches, so a
# 30–80 Hz "gamma" band is half empty and dented by the notch. Default here is
# the honest 30–55 Hz band. To use 30–80 you must re-run preprocessing at
# 1–100 Hz and record the notch stopbands (set EEG_BANDPASS accordingly).
EEG_BANDPASS_HZ = (1.0, 60.0)
EEG_NOTCH_HZ = (60.0, 120.0)
EEG_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 55.0),   # was (30, 80) — see B5
}

# M1: spectral + permutation parameters, surfaced instead of buried.
EEG_WELCH_NPERSEG_SEC = 2.0
EEG_WELCH_OVERLAP = 0.5
EEG_N_PERMUTATIONS = 5000
EEG_CLUSTER_TAIL = 0          # two-sided
EEG_CLUSTER_THRESHOLD = None  # None -> MNE picks the parametric threshold

# Epoch windows over 0-indexed blocks
EEG_EPOCHS = {"early": [0], "mid": [3, 4], "late": [7]}

# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
COLOR_STIM = "#AA4499"
COLOR_SHAM = "#44AA99"
COLOR_STIM_LINE = "#CC79A7"
COLOR_SHAM_LINE = "#009E73"
CONTROL_COLORS = {"LOCNeedle": "#88CCEE", "GSNeedle": "#DDCC77", "Airspeed": "#882255"}

FIG_DPI = 300
SAVE_FORMAT = "svg"


@dataclass
class RunConfig:
    """Bundle passed to the pipeline; every field defaults from the module
    constants above so `RunConfig()` reproduces the reported analysis."""

    paths: DataPaths = field(default_factory=lambda: DATA)
    out_dir: Path = field(default_factory=lambda: OUT_DIR)
    alpha: float = ALPHA
    multiplicity_method: str = MULTIPLICITY_METHOD
    slope_method: str = SLOPE_METHOD
    physio_re_formula: str = PHYSIO_RE_FORMULA
    eeg_bands: dict = field(default_factory=lambda: dict(EEG_BANDS))
    save_figures: bool = True

    def __post_init__(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
