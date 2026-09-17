# Quick_Checklist_by_Action.md — disposition

Three buckets:

* **[CODE]** — implemented in `tvns_sensorimotor`; re-running the pipeline
  produces the corrected numbers/figures.
* **[CODE→TEXT]** — the code now emits the right value or sentence; a human still
  has to paste it into the manuscript.
* **[TEXT]** — manuscript-only (prose, citations, formatting). The exact edit is
  given so the pass is mechanical; nothing in the analysis changes.

Numbers below are from `python -m tvns_sensorimotor.run_all` on
`SDCP_flight-performance-metrics_2023-11-02.xlsx - Cohort2 (1).csv`
(28 subjects, 14/14).

---

## CRITICAL FIXES

### B3 — β coefficient labels  **[CODE→TEXT]**
* `stats_behavior.COEF_LABELS` names the eq.(2) terms:
  `trial_c` = β₁ within-block slope (Sham); `treatment[T.Stim]` = β₂ **mean
  performance-level difference**; `trial_c:treatment[T.Stim]` = β₃ **slope
  difference**. `table1.csv` carries the `coef` column.
* Regenerated Block 1: β₁ = 4.47 (p = 3.5e-4), **β₂ = 12.31** (p = 0.045),
  **β₃ = 2.95** (p = 0.091).
* Manuscript: Abstract — replace "12.438-point-per-trial improvement" with
  "12.4-unit higher mean performance level (β₂)"; drop "per-trial". Title —
  "accelerates" → "is associated with greater early adaptation in" (see B10).
  Table 1 — add the header row `β₁ (trial) | β₂ (treatment, mean level) |
  β₃ (trial×treatment, slope)`.

### B1 — baseline sign convention  **[CODE]**
* `prep.add_baseline_correction`: `improve = baseline_mean − run`; a hard
  assertion checks that block-mean `RMSE_Needle_improve` is predominantly
  positive (matches Figure 2A). `config.IMPROVEMENT_IS_BASELINE_MINUS_RUN`.
* Manuscript §2.5: use the suggested sentence — "each run score was subtracted
  from this baseline value, so a **positive** normalized score indicates
  performance **better** than baseline."

### S1 — Bonferroni across 8 blocks  **[CODE]**
* `fit_block_models` and `compare_block_slopes` add `p_bonf` (per coefficient,
  ×8) and `sig_bonf`.
* Confirmed: Block 1 β₂ **p = 0.045 → p_bonf = 0.36** (no longer significant).
  **No block survives Bonferroni for any Table 1 coefficient.**
* Manuscript: Table 1 significance stars removed except where `sig_bonf`; Methods
  — "Bonferroni correction applied across the eight training blocks."

### B10 — circular slope tests  **[CODE]**
* `stats_behavior.independent_subject_slopes`: one OLS `improve ~ trial_in_block`
  per subject×block, **no treatment term**; group label attached afterwards.
  `compare_block_slopes` runs Welch t-tests + Bonferroni + Cohen's d (also
  satisfies **B9**).
* Regenerated (independent slopes, `table3_slopes.csv`):

  | block | t | df | p | p_bonf | d |
  |--|--|--|--|--|--|
  | 1 | 1.86 | 25.4 | 0.074 | **0.60** | 0.50 |
  | 6 | −1.27 | 23.5 | 0.217 | 1.00 | −0.49 |
  | 8 | −1.26 | 26.0 | 0.218 | 1.00 | −0.48 |

  **No block is significant after correction.** The Block 1 / 6 / 8 differences
  reported from BLUPs do not survive the non-circular test.
* Manuscript: soften the title; in Results state slopes were compared with
  "independent per-subject OLS fits within block (no group term)"; report the
  table above; treat the blockwise trends as descriptive.

### B7 — EEG permutation failure  **[CODE→TEXT]**
* `eeg.run_cluster_tests` returns `n_sig_clusters`; `eeg.summarize_clusters`
  produces: *"Cluster-based permutation testing revealed no statistically
  significant spatiotemporal clusters for any band or training epoch … Figure 5
  differences are descriptive only."* No thresholding trick is used to
  manufacture a cluster.
* Manuscript: Abstract — "EEG spectral analysis revealed elevated frontal gamma
  …" → "Exploratory EEG analysis showed numerically elevated gamma under
  stimulation; no effect survived permutation correction." Results opening —
  paste the sentence above.

### B4 — pupil direction  **[CODE→TEXT]**
* `stats_physio.interpret_slopes` computes Sham slope = β₁, Stim slope = β₁+β₃,
  and prints e.g. *"Under tVNS, Pupil Diameter changes more shallowly (closer to
  zero) than sham (−0.0016 vs −0.0022 per trial)"*.
* Table 2 (manuscript) has Trial β = −0.002, Trial×Treatment β = +0.001 →
  −0.002 + 0.001 = −0.001: **shallower**, not steeper.
* Manuscript: Discussion and Results — "Pupil declined **more slowly (shallower
  slope)** under tVNS."

---

## DATA & METHODS

### S2 — expand Tables 1 & 2  **[CODE]**
`table1.csv` / `table2.csv` columns: `coef, beta, se, ci_low, ci_high, stat,
stat_kind, p, p_bonf, n_obs, n_groups, re_var_intercept, re_var_slope,
ols_fallback`. Blocks 6 and 7 (Needle) hit the **OLS fallback** — flagged, not
silent. Manuscript: add the footnote "OLS fallback used in blocks: 6, 7."

### M1 & M3 — EEG preprocessing methods  **[CODE→TEXT]**
`eeg.describe_preprocessing()` returns the parameter string (Welch nperseg 2 s /
50 % overlap; 5000 permutations; two-sided; montage adjacency; band table). Prose
about reference scheme, ICA algorithm, downsampling, EMG control is
**[TEXT]** — the values aren't recoverable from this repo.

### B5 — EEG filter / gamma incompatibility  **[CODE]**
`config.EEG_BANDS["gamma"] = (30, 55)` with `EEG_BANDPASS_HZ = (1, 60)`,
`EEG_NOTCH_HZ = (60, 120)`. To use 30–80 Hz you must re-preprocess at 1–100 Hz
and set `EEG_BANDPASS_HZ`/notch accordingly — the code will then follow.
Manuscript: redefine gamma as 30–55 Hz (and re-run) **or** state the 1–100 Hz
filter; update Methods + Figure 5 caption + Results.

### M2 — Methods citations  **[TEXT]** MNE-Python (Gramfort 2013), BioSPPy
(Carreiras/Pereira 2015), statsmodels (Seabold & Perktold 2010), gammaCore
(electroCore spec), Maris & Oostenveld (2007), Prepar3D. None are code changes.

### S12 — NaN handling  **[CODE]** `prep.missingness_table(df, measures)` →
`physio_missingness.csv` (n present / missing / % per measure × group). Methods:
"Listwise deletion; per-measure removed counts in Supplementary Table Sx."

### S4 — RMSSD right-skew  **[CODE]** `fit_physio_models` additionally fits
`log(RMSSD) ~ trial_c * treatment` **and** `RMSSD (no >200 ms truncation)` from
`_rmssd_raw`. All three rows land in `table2.csv`; compare the β₃ sign across
them. `config.RMSSD_LOG_MODEL`, `RMSSD_SENSITIVITY_NO_TRUNCATION`.

---

## TEXT & HYPOTHESIS

### S3 — state the failed hypothesis  **[CODE→TEXT]**
`stats_physio.A_PRIORI` encodes "HR ↓, RMSSD ↑ under tVNS"; `interpret_slopes`
prints observed direction beside it. Manuscript: Methods §2.6 — state the
parasympathetic prediction; Results — report the opposite RMSSD/HR direction
plainly, no reframing.

### S10 — cervical-specific framing  **[TEXT]** Intro: note this is cervical
(tcVNS, gammaCore), most cited tVNS literature is auricular, mechanisms differ.
Discussion RMSSD paragraph: "consistent with cervical-specific sympathetic
effects"; cite cervical-tVNS HR↑/RMSSD↓ reports.

### S8 — pupil luminance caveat  **[TEXT]** Methods: "No correction for scene
luminance, room lighting, or gaze position." Limitations: coarse arousal index.
(Recorded in `config`/`stats_physio` docstrings for the analyst.)

### S11 — retention test  **[CODE→TEXT]**
DR1/DR2 are still excluded from the main *training* pipeline
(`config.FOLLOWUP_DAY_PREFIX`), but a dedicated `tvns_sensorimotor.retention`
module now runs the retention test the checklist asked for:
`prep.build_retention_df` (raw + baseline + `%`-improvement per subject per
day, for every day in the study — D00, D01..D04, DR1, DR2 — using the same
"positive = better" sign convention as `_improve` elsewhere:
`(baseline − day) / baseline * 100`), Welch t-tests Stim vs Sham at each DR
day (raw and `%`-improvement, Bonferroni across all rows), and three figures
(raw bar, `%`-improvement bar, full D00→D04→DR trajectory). Run via
`python -m tvns_sensorimotor.retention` or `run_all.run(do_retention=True)`
(the default).

Result on the committed data (21/28 subjects retained at least one DR day,
attrition heavier at DR2): **no Stim-vs-Sham difference at either DR day**, raw
or `%`-improvement (all `p_bonf` ≈ 1). Both groups are roughly 45–70 % *better*
than their D00 baseline at DR1 and DR2 alike (Needle ≈ +65 %, Airspeed ≈ +50 %)
— i.e. training gains are largely retained, equally in both arms. The full
trajectory (Fig., §5c) shows both groups tracking together from D00 through
D04 and into DR1/DR2, with at most a small, non-significant give-back after
training ends.

Manuscript: Discussion → keep future tense only for a *longer-term* retention
question (e.g. weeks/months out); state instead "Both groups retained the
majority of their training gains at the two post-training sessions (DR1, DR2),
with no evidence of a stimulation-group difference in retention (all
Bonferroni-corrected p > .05)." Limitations: persistence beyond the DR1/DR2
window (days) remains untested.

---

## SAMPLE & DESIGN DOCUMENTATION  — all **[TEXT]**

* **M4** randomization mechanism / allocation concealment / stratification.
* **S9** sample-size justification or post-hoc power; "4/28 female (14 %)"
  limitation.
* **M5** eye-tracker sampling rate, calibration, valid-sample %, blink
  threshold/interpolation (Varjo).
* **S6** blinding: were guesses collected? was analysis blinded? visible neck
  contraction as expectancy risk.
* **S5** dose: 5 kHz sine bursts at 25 Hz; intensity setting / titration; sham
  verification.

---

## SUPPLEMENTARY & EXCLUDED

### S13 — airspeed  **[CODE]** `run_all` already emits `table1_airspeed.csv`
(same per-block model for `RMSE_Airspeed`). Drop it into Supplementary Table S1,
or justify exclusion.

### S7 — GSR & respiration  **[TEXT / data]** Not in the provided tables. Either
report RSA alongside RMSSD (needs the raw respiration channel), or document the
exclusion in Limitations.

---

## WRITING & FORMATTING  — all **[TEXT]** except W7

* **W4** "joystick error" → "instrument flight (path-tracking) error";
  RMSE_Needle = Euclidean norm of glideslope + localizer needle deviation.
  (Encoded in `config.NEEDLE_COMPONENTS` / docstrings.)
* **W7** decoding: optional **[CODE]** — add `sklearn.linear_model.
  LogisticRegression` + `roc_auc_score` on `eeg_feats` to predict treatment,
  report AUC in Supplementary; otherwise remove the promise. Not implemented
  (no clear target frame); ~20 lines in `exploratory.py` if wanted.
* **W10** standardize "locus coeruleus–norepinephrine (LC–NE)" (en-dash), define
  once.
* **X2** data-availability statement (OSF/GitHub/"on request").
* **X3** CRediT roles for SK, AN, PB, PS.
* **X5** `\ref{}` all figures/tables; add a Figure 4 callout.
* **X6** Figures 2 & 3 to full text width; labels 10 pt, legend 9 pt.
* **X11** use × (U+00D7) throughout tables.
* **X12** table note: "* p < .05, ** p < .01, *** p < .001 (Bonferroni-
  corrected)".
* **X13** add a Conclusion; keep "Limitations and Future Directions" as a
  subsection.
* **X7/X8** IOP `.bst`; delete superseded figure files.

---

## VALIDATION CHECKS

| check | status |
|---|---|
| all Table 1/2 p-values Bonferroni-corrected | ✅ `p_bonf` columns |
| Block 1 p = 0.044 → 0.35 after correction | ✅ 0.045 → 0.36 |
| slopes from independent OLS within block | ✅ `independent_subject_slopes` |
| EEG filter ≥ 60 Hz vs gamma, or gamma = 30–55 | ✅ gamma = 30–55 default |
| permutation results reported honestly | ✅ `summarize_clusters` |
| pupil text corrected (shallower) | ✅ `interpret_slopes` |
| RMSSD framed cervical-specific | ⬜ TEXT |
| figures cited with `\ref{}` | ⬜ TEXT |
| tables have full headers/row labels | ✅ CSV / ⬜ TEXT layout |
| Methods ≥ 1 citation per paragraph | ⬜ TEXT |
| parasympathetic hypothesis stated + marked unsupported | ⬜ TEXT (code prints direction) |
| Limitations: blinding, gender, sample size | ⬜ TEXT |
| retention (DR-day) test run, Stim vs Sham | ✅ `retention` module — no group difference found |
