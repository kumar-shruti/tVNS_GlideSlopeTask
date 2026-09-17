"""
Exploratory / not-reported analyses.

These are kept for provenance from the original notebook but are **not** part of
the manuscript pipeline and were not corrected for the checklist. Do not cite
numbers from here without re-running them through the same scrutiny as the
primary analyses.

Contents (as callables, each takes `trials_df` / `physio_df` and returns a
Figure or DataFrame):

  waterfall_early_delta          per-subject Δ(block 1–2 − baseline) bar chart
  baseline_vs_delta              baseline level vs early improvement scatter
  power_law_fits                 c + a·trial^b curve fits per subject/measure
  within_block_gain_boxplots     early-3 vs late-3 within-block "gain"
  rmse_tradeoff                  needle vs airspeed correlation structure
  hrv_pca_performance            PCA on 7 HRV features -> RMSE mixed models
  gam_pc_learning                per-condition GAM of PC vs improvement
  rf_pc_learning                 random-forest PC -> improvement (CV R²)
  eeg_power_trajectory_darkstyle the styled dark trajectory panel

Why they were dropped from the main path:
  * abandoned modelling directions (GAM, RF, power-law) with no inferential
    payoff and small per-cell n;
  * `block = trial // 6` redefinitions that conflict with the manuscript's
    day-anchored block rule;
  * duplicate correlation-heatmap variants;
  * plots superseded by Figures 1–5.

Port them here on demand — this module is intentionally a thin placeholder so
the primary package stays legible.
"""
from __future__ import annotations

__all__: list[str] = []
