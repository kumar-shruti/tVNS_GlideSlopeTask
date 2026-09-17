"""
Data loading. One function per source; each returns a tidy DataFrame
Everything is a local file path resolved through `config.DATA`.
"""
from __future__ import annotations

import pickle
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Flight performance                                                           #
# --------------------------------------------------------------------------- #
_PERF_DROP_COLS = [
    "date", "time", "total_time", "removal_category", "remove_reasoning",
    "Crosses", "Mean", "SD", "Var", "MAD", "MSE", "Norm",
]


def load_performance(path=None, keep_followup: bool = False) -> pd.DataFrame:
    """Long-format flight metrics, one row per subject × run × measure.

    Applies the study's exclusions exactly once:
      * ``remove == True`` rows are dropped (curator flag on the source sheet).
      * retention days (``DR1``/``DR2``) are dropped from the main (training)
        analyses unless ``keep_followup=True`` — set that for the retention
        module (:mod:`tvns_sensorimotor.retention`), which is the only caller
        that needs the DR days.

    ``day_num`` is 0..4 for D00..D04. Retention days don't fit that scheme (the
    "R" breaks a plain ``int(day[1:])``), so with ``keep_followup=True`` they
    get the disjoint sentinel ``100 + n`` (DR1 -> 101, DR2 -> 102, ...) instead
    of NaN — a NaN index key would silently vanish under ``pivot_table``.
    """
    path = config.DATA.require("performance") if path is None else path
    df = pd.read_csv(path)

    df = df[df["remove"] != True].copy()  # noqa: E712  (curator flag)
    df = df.drop(columns=[c for c in _PERF_DROP_COLS + ["remove"] if c in df.columns])

    for c in ("subject", "day", "run"):
        df[c] = df[c].astype(str)

    is_retention = df["day"].str.startswith(config.FOLLOWUP_DAY_PREFIX)
    if not keep_followup:
        df = df[~is_retention].copy()
        is_retention = is_retention[~is_retention]  # now empty/all-False, kept in sync

    df["run_num"] = df["run"].str.replace("R", "", regex=False).astype(int)

    df["day_num"] = 0
    df.loc[~is_retention, "day_num"] = (
        df.loc[~is_retention, "day"].str.replace("D", "", regex=False).astype(int)
    )
    if is_retention.any():
        df.loc[is_retention, "day_num"] = 100 + (
            df.loc[is_retention, "day"]
            .str.replace(config.FOLLOWUP_DAY_PREFIX, "", regex=False)
            .astype(int)
        )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Derived ECG + pupil table                                                    #
# --------------------------------------------------------------------------- #
_PHYSIO_RENAME = {"Subject": "subject", "Day": "day", "Run": "run"}


def load_physio(path=None) -> pd.DataFrame:
    """Run-level ECG + pupil table (``final_physio_table.csv``).

    Kept close to source: renames the id columns, drops the 'Percent Complete
    Data' column if present, and returns raw values. Plausibility masking and
    baseline correction live in :mod:`tvns_sensorimotor.prep` so the choices are
    explicit and testable (checklist S8, S12, B1).
    """
    path = config.DATA.require("physio") if path is None else path
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["Percent Complete Data"] if c in df.columns])
    df = df.rename(columns=_PHYSIO_RENAME)
    for c in ("subject", "day", "run"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def load_eeg_features(path=None) -> pd.DataFrame:
    """Trial-level EEG spectral features (``eeg_feats.csv``)."""
    path = config.DATA.require("eeg_features") if path is None else path
    df = pd.read_csv(path, index_col=0)
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Raw epoched EEG                                                              #
# --------------------------------------------------------------------------- #
class RawEEG(NamedTuple):
    X: np.ndarray          # (n_epochs, n_channels, n_times)
    y: np.ndarray          # performance target (unused here)
    treatment: np.ndarray  # (n_epochs,)
    blocks: np.ndarray     # (n_epochs,) block index as stored ("trials" in the pickle)
    subject: np.ndarray    # (n_epochs,)


def load_eeg_raw(path=None, n_channels: int | None = None) -> RawEEG:
    """Load ``nn_data.pkl`` -> (X, y, treatment, trials, subj_ids).

    The pickle's 4th element is the block index (named 'trials' in the original
    script). We keep the first ``n_channels`` channels (the montage-mapped EEG;
    trailing rows are EOG/EXG).
    """
    path = config.DATA.require("eeg_raw") if path is None else path
    n_channels = config.EEG_N_CHANNELS if n_channels is None else n_channels
    with open(path, "rb") as fh:
        X, y, treatment, trials, subj_ids = pickle.load(fh)
    X = np.asarray(X, dtype=np.float64)[:, :n_channels, :]
    return RawEEG(
        X=X,
        y=np.asarray(y),
        treatment=np.asarray(treatment).ravel(),
        blocks=np.asarray(trials).ravel(),
        subject=np.asarray(subj_ids).ravel(),
    )
