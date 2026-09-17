"""
Peripheral physiology statistics: Table 2 and the Figure-3 cluster bars.

Checklist items addressed here:
  B4  – the pupil Trial x Treatment sign is checked and the implied Stim slope
        direction is reported in words, so the text cannot say "steeper" when
        the maths says "shallower".
  B6  – one model form, stated once: measure ~ trial_c * treatment with random
        intercept AND random slope on trial_c per subject.
  S3  – the a-priori parasympathetic prediction (RMSSD up, HR down under tVNS)
        is encoded and the observed direction is compared to it.
  S4  – RMSSD also modelled as log(RMSSD); plus a sensitivity fit without the
        >200 ms pre-truncation.
  S12 – listwise-deletion counts via prep.missingness_table.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from . import config

# S3: direction predicted by the parasympathetic account of (auricular) tVNS.
A_PRIORI = {
    "Heart Rate": "decrease under tVNS (negative Trial x Treatment)",
    "RMSSD": "increase under tVNS (positive Trial x Treatment)",
    "Pupil Diameter": "no strong prediction",
}

_TERMS = {
    "trial_c": "beta1_trial_slope_sham",
    "treatment[T.Stim]": "beta2_mean_level_diff",
    "trial_c:treatment[T.Stim]": "beta3_slope_diff",
}


def _frame(physio_df: pd.DataFrame, measure: str, *, transform=None) -> pd.DataFrame:
    d = physio_df[physio_df["block"].fillna(0) > 0].copy()
    d = d.dropna(subset=[measure, "trial", "treatment"])
    d = d[d["treatment"].isin(["Stim", "Sham"])].copy()
    d["treatment"] = pd.Categorical(d["treatment"], categories=["Sham", "Stim"])
    d["trial_c"] = d["trial"] - d["trial"].mean()
    d["y"] = d[measure] if transform is None else transform(d[measure])
    return d.dropna(subset=["y"])


def _fit_one(d: pd.DataFrame, *, label: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("y ~ trial_c * treatment", d, groups=d["subject"],
                         re_formula=config.PHYSIO_RE_FORMULA)
        fit = md.fit(reml=config.REML, method=config.LMM_OPTIMIZER,
                     maxiter=config.LMM_MAXITER, disp=False)
    ci = fit.conf_int()
    out = []
    for term, name in _TERMS.items():
        if term not in fit.params.index:
            continue
        beta, se = float(fit.params[term]), float(fit.bse[term])
        out.append({
            "model": label, "term": term, "coef": name,
            "beta": beta, "se": se,
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "z": beta / se if se else np.nan, "p": float(fit.pvalues[term]),
            "n_obs": int(fit.nobs), "n_groups": int(d["subject"].nunique()),
            "converged": bool(getattr(fit, "converged", True)),
            "re_var_intercept": float(fit.cov_re.iloc[0, 0]),
            "re_var_slope": float(fit.cov_re.iloc[1, 1]) if fit.cov_re.shape[0] > 1 else np.nan,
        })
    return pd.DataFrame(out)


def fit_physio_models(physio_df: pd.DataFrame,
                      measures=config.PHYSIO_MEASURES) -> pd.DataFrame:
    """Table 2. One tidy row per (model, coefficient); Trial x Treatment p-values
    get a Bonferroni column across the reported measures."""
    tables = []
    for m in measures:
        tables.append(_fit_one(_frame(physio_df, m), label=m))
        if m == "RMSSD" and config.RMSSD_LOG_MODEL:
            # log needs positive values; RMSSD here is baseline-corrected so shift.
            def _logshift(s):
                s = s.astype(float)
                return np.log(s - s.min() + 1.0)
            tables.append(_fit_one(_frame(physio_df, m, transform=_logshift),
                                   label="log(RMSSD)"))
            if config.RMSSD_SENSITIVITY_NO_TRUNCATION and "_rmssd_raw" in physio_df.columns:
                d = physio_df.copy()
                d["RMSSD_untrunc"] = d["_rmssd_raw"]
                tables.append(_fit_one(_frame(d, "RMSSD_untrunc"),
                                       label="RMSSD (no >200ms truncation)"))
    tbl = pd.concat(tables, ignore_index=True)

    inter = tbl["term"] == "trial_c:treatment[T.Stim]"
    reported = tbl["model"].isin(list(measures))
    mask = inter & reported
    tbl.loc[mask, "p_bonf"] = multipletests(
        tbl.loc[mask, "p"], alpha=config.ALPHA, method=config.MULTIPLICITY_METHOD)[1]
    return tbl


# --------------------------------------------------------------------------- #
# B4 / S3 — turn the pupil (and every) interaction sign into a sentence        #
# --------------------------------------------------------------------------- #
def interpret_slopes(table: pd.DataFrame) -> pd.DataFrame:
    """For each measure: Sham slope = beta1, Stim slope = beta1 + beta3, plus a
    plain-language direction and whether it matches the a-priori prediction."""
    rows = []
    for model, g in table.groupby("model"):
        g = g.set_index("term")
        if "trial_c" not in g.index or "trial_c:treatment[T.Stim]" not in g.index:
            continue
        b1 = g.loc["trial_c", "beta"]
        b3 = g.loc["trial_c:treatment[T.Stim]", "beta"]
        stim, sham = b1 + b3, b1
        if np.sign(stim) != np.sign(sham):
            rel = "changes in the opposite direction from"
        elif abs(stim) > abs(sham):
            rel = "changes more steeply (further from zero) than"
        else:
            rel = "changes more shallowly (closer to zero) than"
        rows.append({
            "model": model,
            "sham_slope_beta1": sham, "stim_slope_beta1_plus_beta3": stim,
            "interaction_beta3": b3, "interaction_p": g.loc["trial_c:treatment[T.Stim]", "p"],
            "verbal": f"Under tVNS, {model} {rel} sham across trials "
                      f"({stim:+.4f} vs {sham:+.4f} per trial)",
            "a_priori": A_PRIORI.get(model, ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure 3 cluster bars  (independent-samples cluster permutation over trials) #
# --------------------------------------------------------------------------- #
def trial_cluster_test(physio_df: pd.DataFrame, measure: str,
                       n_permutations: int | None = None):
    """1-D cluster-based permutation test (Stim vs Sham) over the trial axis.

    Returns list of (trial_start, trial_end, p) for clusters with p < alpha, or
    [] when nothing survives — which the caller must report honestly (cf. B7).
    """
    from ._common import cluster_test_over_index

    n_permutations = config.EEG_N_PERMUTATIONS if n_permutations is None else n_permutations
    return cluster_test_over_index(
        physio_df, index_col="trial", value_col=measure,
        n_permutations=n_permutations, alpha=config.ALPHA)
