"""
tvns_sensorimotor
=================

Analysis pipeline for the arousal-linked tVNS sensorimotor-adaptation study,
refactored from ``tVNS_sensorimotor_analysis copy.ipynb``.

Layout
------
config            paths, constants, colours, and every checklist knob
io                load raw tables / pickle (no stats)
prep              block/trial labels, needle composite, baseline correction,
                  physio cleaning, arousal merge, missingness table
stats_behavior    Table 1 (per-block LMM) + Table 3 (independent slope test)
stats_physio      Table 2 (physio LMM, log-RMSSD, sensitivity) + cluster bars
cluster_permutation  1-D cluster-permutation test (Stim vs Sham), RMSE_Needle
                  / RMSE_Airspeed across the training trial axis
eeg               band power + cluster-permutation tests (Figure 5)
arousal           PCA arousal latent variable (Figure 4)
retention         post-training (DR-day) retention: raw + % change + trajectory
figures           the four manuscript figures
exploratory       everything the notebook tried but did not report
run_all           one call that regenerates every table + figure

Typical use
-----------
>>> from tvns_sensorimotor import run_all
>>> res = run_all.run(save=True)
>>> res["table1"].head()
"""
from . import (config, io, prep, stats_behavior, stats_physio, cluster_permutation,
              eeg, arousal, retention, figures)  # noqa: F401

__all__ = ["config", "io", "prep", "stats_behavior", "stats_physio", "cluster_permutation",
           "eeg", "arousal", "retention", "figures", "run_all"]

__version__ = "1.0.0"
