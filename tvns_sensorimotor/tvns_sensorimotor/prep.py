"""
Turn the raw long tables into the two analysis frames used everywhere else:

    trials_df   one row per subject × run (wide over measures), with block /
                trial-in-block labels, per-subject baseline, and baseline-
                corrected "improvement" columns.

    physio_df   trials_df left-joined to run-level ECG/pupil, with plausibility
                masking and baseline correction applied explicitly.

    retention_df   one row per subject × day, for D00 (baseline) and every
                post-training retention day (DR1, DR2, ...): mean RMSE for that
                day plus its baseline value and % change from baseline. Used by
                :mod:`tvns_sensorimotor.retention`.

All the block/trial bookkeeping that the notebook repeated (and sometimes
redefined inconsistently) lives here once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Block / trial labelling                                                      #
# --------------------------------------------------------------------------- #
def add_block_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``block`` (0 = baseline, 1..8 training) and a continuous ``trial``.

    Block rule (the manuscript's): each training day contributes two blocks,
    runs 1–6 and runs 7–12. Baseline day -> block 0.
    """
    df = df.copy()
    df["block"] = np.nan
    df.loc[df["day_num"] == config.BASELINE_DAY, "block"] = 0

    counter = 1
    for day in sorted(d for d in df["day_num"].unique() if d > 0):
        m = df["day_num"] == day
        df.loc[m & df["run_num"].between(1, 6), "block"] = counter
        df.loc[m & df["run_num"].between(7, 12), "block"] = counter + 1
        counter += 2
    df["block"] = df["block"].astype("Int64")

    # Continuous learning index. Baseline runs are 1..3; training runs continue
    # from 4 (run_num + (day_num-1)*12). NB: the legacy notebook used "+2" here,
    # which put the first training trial at 3 and overlapped baseline; that only
    # ever affected the x-axis of the learning-curve plots.
    df["trial"] = df["run_num"]
    train = df["day_num"] > 0
    df.loc[train, "trial"] = df.loc[train, "run_num"] + (df.loc[train, "day_num"] - 1) * 12 + 3

    # 1..RUNS_PER_BLOCK position inside each block (used by every slope model).
    df["trial_in_block"] = (
        df.sort_values(["subject", "block", "trial"])
        .groupby(["subject", "block"])
        .cumcount()
        + 1
    )
    df.loc[df["block"] == 0, "trial_in_block"] = pd.NA
    return df


# --------------------------------------------------------------------------- #
# Pivot long -> wide over measures, build the needle composite                 #
# --------------------------------------------------------------------------- #
def pivot_measures(long_df: pd.DataFrame) -> pd.DataFrame:
    idx = ["subject", "treatment", "day", "run", "task",
           "run_num", "day_num"]
    wide = (
        long_df.pivot_table(index=idx, columns="measure", values="RMSE")
        .rename(columns=lambda m: f"RMSE_{m}")
        .reset_index()
    )
    gs, loc = config.NEEDLE_COMPONENTS
    wide["RMSE_Needle"] = np.sqrt(wide[gs] ** 2 + wide[loc] ** 2)
    return wide


# --------------------------------------------------------------------------- #
# Baseline correction  (checklist B1)                                          #
# --------------------------------------------------------------------------- #
def add_baseline_correction(
    df: pd.DataFrame,
    measures=config.BEHAVIOR_MEASURES,
    verify: bool = True,
) -> pd.DataFrame:
    """Per-subject Day-0 mean, then ``improvement = baseline_mean - run`` and
    ``pct_change = improvement / baseline_mean * 100``.

    POSITIVE improvement/pct_change == performed better than baseline (B1),
    the same convention `build_retention_df` uses for its day-level
    ``_pct_change`` columns (here at trial granularity). With `verify` the
    block-mean improvement is checked to be predominantly >= 0, matching
    Figure 2A.
    """
    if not config.IMPROVEMENT_IS_BASELINE_MINUS_RUN:  # pragma: no cover
        raise ValueError("Only the baseline-minus-run convention is supported (B1).")

    df = df.copy()
    base = (
        df[df["day_num"] == config.BASELINE_DAY]
        .groupby("subject")[list(measures)]
        .mean()
        .rename(columns={m: f"{m}_baseline" for m in measures})
        .reset_index()
    )
    df = df.merge(base, on="subject", how="left")
    for m in measures:
        df[f"{m}_improve"] = df[f"{m}_baseline"] - df[m]
        df[f"{m}_pct_change"] = df[f"{m}_improve"] / df[f"{m}_baseline"] * 100

    if verify:
        chk = (
            df[df["block"] > 0]
            .groupby("block")["RMSE_Needle_improve"]
            .mean()
        )
        frac_pos = (chk > 0).mean()
        if frac_pos < 0.5:
            raise AssertionError(
                "B1 sign check failed: most block-mean RMSE_Needle_improve values "
                f"are negative ({frac_pos:.0%} positive). Improvement should be "
                "baseline - run (positive = better)."
            )
    return df


def build_trials_df(long_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """End-to-end behaviour frame: load -> label -> pivot -> baseline-correct."""
    from . import io

    long_df = io.load_performance() if long_df is None else long_df
    long_df = add_block_labels(long_df)
    wide = pivot_measures(long_df)
    wide = add_block_labels(wide)  # re-derive on the wide frame (labels dropped by pivot)
    wide = add_baseline_correction(wide)
    return wide


# --------------------------------------------------------------------------- #
# Physio                                                                       #
# --------------------------------------------------------------------------- #
def clean_physio(physio_raw: pd.DataFrame) -> pd.DataFrame:
    """Plausibility masking (S8) + per-subject baseline correction (B1).

    * Heart Rate outside HR_PLAUSIBLE_BPM -> NaN (HR and RMSSD both dropped for
      that run, since an implausible HR invalidates the RR series).
    * Pupil outside PUPIL_PLAUSIBLE_MM -> NaN.
    * RMSSD legacy truncation is applied here but recorded; the sensitivity
      model in stats.physio can re-run without it (S4).
    * Baseline = Day-0 mean per subject; fallback = Day-1 first practice (PBL01)
      run when Day-0 is missing (e.g. P416).
    """
    df = physio_raw.copy()
    lo, hi = config.HR_PLAUSIBLE_BPM
    bad_hr = df["Heart Rate"].lt(lo) | df["Heart Rate"].gt(hi)
    df.loc[bad_hr, ["Heart Rate", "RMSSD"]] = np.nan

    if "Pupil Diameter" in df.columns:
        plo, phi = config.PUPIL_PLAUSIBLE_MM
        df.loc[df["Pupil Diameter"].lt(plo) | df["Pupil Diameter"].gt(phi), "Pupil Diameter"] = np.nan

    df["_rmssd_raw"] = df["RMSSD"]  # keep for the no-truncation sensitivity fit
    rlo, rhi = config.RMSSD_TRUNCATION_MS
    df.loc[df["RMSSD"].lt(rlo) | df["RMSSD"].gt(rhi), "RMSSD"] = np.nan

    corr_cols = [c for c in ("Heart Rate", "RMSSD", "Pupil Diameter") if c in df.columns]
    is_base = df["day"].eq("D00")
    is_pbl1 = df["day"].eq("D01") & df["run"].astype(str).str.upper().eq("PBL01")
    base = df[is_base].groupby("subject")[corr_cols].mean()
    fallback = df[is_pbl1].groupby("subject")[corr_cols].mean()
    base = base.reindex(df["subject"].unique()).fillna(fallback)

    out = df.copy()
    for subj, row in base.iterrows():
        m = out["subject"].eq(subj)
        out.loc[m, corr_cols] = out.loc[m, corr_cols].to_numpy() - row[corr_cols].to_numpy()

    # drop baseline/practice rows from the analysis frame
    out = out[~out["run"].astype(str).str.upper().str.contains("PBL")]
    out = out[out["day"].ne("D00")].drop_duplicates()
    return out


def build_physio_df(trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """trials_df joined to cleaned run-level physio (run-level values repeat
    across the runs' trials — physio is sampled once per run)."""
    from . import io

    trials_df = build_trials_df() if trials_df is None else trials_df
    physio = clean_physio(io.load_physio())
    keep = ["subject", "day", "run", "Heart Rate", "RMSSD", "_rmssd_raw"]
    keep += [c for c in ["Pupil Diameter"] if c in physio.columns]
    merged = trials_df.merge(physio[keep], on=["subject", "day", "run"], how="left")
    return merged


def build_arousal_df(physio_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """physio_df + trial-level EEG features, with the lower-case column names
    the PCA in :mod:`tvns_sensorimotor.arousal` expects."""
    from . import io

    physio_df = build_physio_df() if physio_df is None else physio_df
    d = physio_df.rename(columns={
        "Heart Rate": "heart_rate", "RMSSD": "rmssd", "Pupil Diameter": "pupil",
    })
    eeg = io.load_eeg_features()
    eeg_cols = [c for c in ["subject", "trial", "treatment",
                            "frontocentral_theta", "posterior_alpha"] if c in eeg.columns]
    return d.merge(eeg[eeg_cols], on=[c for c in ["subject", "trial", "treatment"]
                                      if c in eeg_cols], how="left")


# --------------------------------------------------------------------------- #
# Retention (post-training DR days)                                            #
# --------------------------------------------------------------------------- #
def _baseline_day_label() -> str:
    return f"D{config.BASELINE_DAY:02d}"


def build_retention_df(measures=config.BEHAVIOR_MEASURES) -> pd.DataFrame:
    """One row per subject × day, for every day in the study: D00 (baseline),
    every training day (D01..D04), and every retention day (DR1, DR2, ...).

    Each day's runs are averaged to one value per subject (12 runs/day, same
    for a training day, a retention day, or — with only 3 runs — D00), then
    baseline-normalized two ways, in the same "positive = better" direction as
    `add_baseline_correction`'s ``_improve`` columns elsewhere in the package:

      ``{measure}``               raw day mean
      ``{measure}_baseline``      that subject's D00 mean
      ``{measure}_pct_change``    (baseline − day) / baseline * 100
                                   (0 by construction for D00 itself;
                                   positive = lower RMSE than baseline, i.e.
                                   improved; negative = worse than baseline)

    ``day_order`` gives a numeric x-axis that runs D00 -> 0, D01 -> 1, ...,
    D04 -> 4, then continues into the retention days: DR1 -> 5, DR2 -> 6, ...
    """
    from . import io

    raw = io.load_performance(keep_followup=True)
    per_day = pivot_measures(raw)  # every day present in the source CSV: D00..D04, DR1, DR2

    base_label = _baseline_day_label()
    measures = list(measures)
    per_day = per_day.groupby(["subject", "treatment", "day"], as_index=False)[measures].mean()

    baseline = (
        per_day.loc[per_day["day"].eq(base_label)]
        .set_index("subject")[measures]
        .add_suffix("_baseline")
    )
    per_day = per_day.join(baseline, on="subject")

    pct_cols = [f"{m}_pct_change" for m in measures]
    for m, pct_col in zip(measures, pct_cols):
        per_day[pct_col] = (per_day[f"{m}_baseline"] - per_day[m]) / per_day[f"{m}_baseline"] * 100
    per_day.loc[per_day["day"].eq(base_label), pct_cols] = 0.0

    is_retention = per_day["day"].str.startswith(config.FOLLOWUP_DAY_PREFIX)
    last_training_day = max(
        int(day.replace("D", "")) for day in per_day.loc[~is_retention, "day"].unique()
    )
    order = {day: int(day.replace("D", "")) for day in per_day.loc[~is_retention, "day"].unique()}
    for day in sorted(per_day.loc[is_retention, "day"].unique()):
        order[day] = last_training_day + int(day.replace(config.FOLLOWUP_DAY_PREFIX, ""))
    per_day["day_order"] = per_day["day"].map(order)

    return per_day.sort_values(["subject", "day_order"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Missingness report  (checklist S12)                                          #
# --------------------------------------------------------------------------- #
def missingness_table(df: pd.DataFrame, measures, by="treatment") -> pd.DataFrame:
    """n present / n missing per measure per group, for the Methods supplement."""
    rows = []
    post = df[df["block"] > 0]
    for m in measures:
        for grp, g in post.groupby(by):
            rows.append({
                by: grp, "measure": m,
                "n_present": int(g[m].notna().sum()),
                "n_missing": int(g[m].isna().sum()),
                "pct_missing": round(100 * g[m].isna().mean(), 1),
            })
    return pd.DataFrame(rows).sort_values(["measure", by]).reset_index(drop=True)
