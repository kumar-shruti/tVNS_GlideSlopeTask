# tvns_sensorimotor

Refactor of `tVNS_sensorimotor_analysis copy.ipynb` (175 cells, ~250 KB of
source, baseline/​block/​`bs_df` recomputed a dozen times, three mutually
inconsistent `block` definitions, abandoned GAM/​RandomForest/​power-law
branches) into a small, linear, testable package — with the
`Quick_Checklist_by_Action.md` fixes built into the code.

## Install / run

```bash
cd "AFRL tVNS/tvns_sensorimotor"
pip install pandas numpy scipy statsmodels scikit-learn matplotlib mne
python -m tvns_sensorimotor.run_all                # behaviour + cluster test + retention (Tables 1 & 3, Figs 1-2, DR-day analysis)
python -m tvns_sensorimotor.run_all --all          # + physio, arousal, EEG
python -m tvns_sensorimotor.run_all --no-retention # skip retention
python -m tvns_sensorimotor.run_all --no-cluster   # skip the cluster-permutation test

python -m tvns_sensorimotor.retention              # just the retention (DR-day) analysis
python -m tvns_sensorimotor.cluster_permutation    # just the behaviour cluster-permutation test
```

Outputs (CSV tables + SVG figures) land in `outputs/` (override with
`TVNS_OUT_DIR`). Input paths are in `tvns_sensorimotor/config.py`; override the
whole set with `TVNS_DATA_DIR` or by reassigning `config.DATA`.

The committed dataset only includes the flight-performance CSV and the raw EEG
pickle. `final_physio_table.csv` (ECG+pupil) and `eeg_feats.csv` (trial-level
EEG features) are **not** here — the physio / arousal stages skip themselves with
a clear message until you point `config.DATA` at your copies.

## Module map

| module | contents |
|---|---|
| `config` | paths, experiment constants, colours, **every checklist knob** |
| `io` | load CSVs / pickle — no statistics, no Colab |
| `prep` | block+trial labels, needle composite, baseline correction (B1), physio cleaning, arousal merge, missingness table (S12) |
| `stats_behavior` | Table 1 (per-block LMM, full S2 columns, S1 Bonferroni) + Table 3 (independent-slope test, B10/B9) + baseline equivalence + `fit_overall_models` (AIC/BIC across a 9-spec pooled random/fixed-effects ladder) |
| `stats_physio` | Table 2 (B6 model form, S4 log-RMSSD + sensitivity), pupil-direction interpreter (B4), 1-D cluster test |
| `cluster_permutation` | 1-D cluster-permutation test (Stim vs Sham) on RMSE_Needle / RMSE_Airspeed across the training trial axis, honest cluster summary + overlay figure, plus raw uncorrected per-trial Welch t-tests (`pointwise_ttests`) for inspection |
| `eeg` | Welch band power, spatio-temporal cluster permutation, honest cluster summary (B5, B7, M1) |
| `arousal` | within-subject PCA arousal latent variable → learning (Figure 4) |
| `retention` | post-training retention: raw scores and % improvement from baseline at each DR day (Stim vs Sham, S11), plus the full D00→D04→DR trajectory |
| `figures` | the four manuscript figures, plus a % improvement variant of Figure 1 |
| `exploratory` | index + rationale for everything dropped from the reported path |
| `run_all` | regenerate every table + figure |

## What changed vs. the notebook

* One import block, one config object — imports and constants no longer repeated
  per cell.
* `pivot_df` / `bs_df` / `baseline_df` → a single `trials_df` built once by
  `prep.build_trials_df()`; physio adds `physio_df`.
* `block` is defined exactly once (day-anchored, runs 1–6 / 7–12). The
  `block = trial // 6` scratch variants are gone.
* Colab `drive.mount`, `gspread`, and `pip install` cells removed; all inputs
  are local paths.
* Exploratory analyses (waterfall, baseline-vs-delta, power-law, GAM, RF, 3-D
  PCA, per-subject correlation heatmaps) are not in the pipeline — see
  `exploratory.py` for the list and why.

See `CHECKLIST_RESPONSE.md` for the item-by-item disposition of
`Quick_Checklist_by_Action.md`.
