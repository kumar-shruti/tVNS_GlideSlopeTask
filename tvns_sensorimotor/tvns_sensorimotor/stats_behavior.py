"""
Behavioural statistics: Table 1 (per-block mixed models), Table 3 (between-
group comparison of within-block learning slopes), and an overall-model
random-effects/fixed-effects comparison via AIC/BIC (`fit_overall_models`).

Checklist items addressed here:
  B3  – coefficients are labelled by meaning, not position.
  S1  – Bonferroni across the 8 blocks, per coefficient.
  S2  – full coefficient tables: beta, SE, 95% CI, stat, p, p_bonf, n, groups,
        RE variances, and an `ols_fallback` flag.
  B10 – slope comparison uses independent per-subject OLS slopes (no treatment
        term), not BLUPs recovered from a treatment-containing model.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import linregress, ttest_ind
from statsmodels.stats.multitest import multipletests

from . import config
from ._common import cohens_d, welch_df

# Human-readable names for the eq.(2) coefficients (B3).
COEF_LABELS = {
    "Intercept": "beta0_intercept",
    "trial_c": "beta1_within_block_slope_sham",
    "treatment[T.Stim]": "beta2_mean_level_diff",
    "trial_c:treatment[T.Stim]": "beta3_slope_diff_stim_minus_sham",
}


def _prep_block_frame(trials_df: pd.DataFrame, measure: str) -> pd.DataFrame:
    col = f"{measure}_improve"
    d = trials_df[trials_df["block"].fillna(0) > 0].copy()
    d = d.dropna(subset=[col, "trial_in_block", "treatment"])
    d = d[d["treatment"].isin(["Stim", "Sham"])].copy()
    d["treatment"] = pd.Categorical(d["treatment"], categories=["Sham", "Stim"])
    # center trial-in-block so the intercept is mid-block and beta2 is a level diff
    d["trial_c"] = d["trial_in_block"] - d.groupby("block")["trial_in_block"].transform("mean")
    d["y"] = d[col]
    return d


def _coef_row(fit, term, *, is_mixed, n_obs, n_groups, block, measure, ols_fallback, re_var):
    params, bse, pvals = fit.params, fit.bse, fit.pvalues
    ci = fit.conf_int()
    if term not in params.index:
        return None
    beta = float(params[term])
    se = float(bse[term])
    stat = beta / se if se else np.nan
    return {
        "measure": measure,
        "block": int(block),
        "term": term,
        "coef": COEF_LABELS.get(term, term),
        "beta": beta,
        "se": se,
        "ci_low": float(ci.loc[term, 0]),
        "ci_high": float(ci.loc[term, 1]),
        "stat": stat,                       # Wald z (mixed) / t (OLS)
        "stat_kind": "z" if is_mixed else "t",
        "p": float(pvals[term]),
        "n_obs": int(n_obs),
        "n_groups": int(n_groups),
        "ols_fallback": bool(ols_fallback),
        "re_var_intercept": re_var.get("intercept", np.nan),
        "re_var_slope": re_var.get("slope", np.nan),
    }


def fit_block_models(trials_df: pd.DataFrame, measure: str = "RMSE_Needle") -> pd.DataFrame:
    """Per-block:  improve ~ trial_c * treatment  + (1 + trial_c | subject).

    OLS with the same fixed structure is the documented fallback when the mixed
    model will not converge. Returns one row per (block, coefficient) with the
    S2 columns; Bonferroni p-values are added per coefficient across blocks (S1).
    """
    d = _prep_block_frame(trials_df, measure)
    rows = []
    for block, g in d.groupby("block"):
        ols_fallback = False
        re_var = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm("y ~ trial_c * treatment", g,
                                 groups=g["subject"], re_formula="~trial_c")
                fit = md.fit(reml=config.REML, method=config.LMM_OPTIMIZER,
                             maxiter=config.LMM_MAXITER, disp=False)
            if not getattr(fit, "converged", True):
                raise RuntimeError("mixed model did not converge")
            cov_re = fit.cov_re
            re_var = {"intercept": float(cov_re.iloc[0, 0]),
                      "slope": float(cov_re.iloc[1, 1]) if cov_re.shape[0] > 1 else np.nan}
            is_mixed, n_groups = True, g["subject"].nunique()
        except Exception:
            ols_fallback = True
            fit = smf.ols("y ~ trial_c * treatment", data=g).fit()
            is_mixed, n_groups = False, g["subject"].nunique()

        for term in ["Intercept", "trial_c", "treatment[T.Stim]", "trial_c:treatment[T.Stim]"]:
            row = _coef_row(fit, term, is_mixed=is_mixed, n_obs=len(g), n_groups=n_groups,
                            block=block, measure=measure, ols_fallback=ols_fallback, re_var=re_var)
            if row:
                rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl["p_bonf"] = np.nan
    for _term, idx in tbl.groupby("term").groups.items():
        tbl.loc[idx, "p_bonf"] = multipletests(
            tbl.loc[idx, "p"], alpha=config.ALPHA, method=config.MULTIPLICITY_METHOD
        )[1]
    tbl["sig_bonf"] = tbl["p_bonf"] < config.ALPHA
    return tbl.sort_values(["block", "term"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Table 3 — independent within-block slopes  (B10)                             #
# --------------------------------------------------------------------------- #
def independent_subject_slopes(trials_df: pd.DataFrame, measure: str = "RMSE_Needle") -> pd.DataFrame:
    """One OLS slope of ``improve ~ trial_in_block`` per subject × block.

    No treatment term anywhere — the group label is only attached afterwards, so
    the subsequent Stim-vs-Sham test is not circular.
    """
    col = f"{measure}_improve"
    d = trials_df[trials_df["block"].fillna(0) > 0].dropna(subset=[col, "trial_in_block"])
    rows = []
    for (subj, block), g in d.groupby(["subject", "block"]):
        if g["trial_in_block"].nunique() < config.MIN_TRIALS_FOR_SLOPE:
            continue
        lr = linregress(g["trial_in_block"].to_numpy(float), g[col].to_numpy(float))
        rows.append({"subject": subj, "block": int(block),
                     "treatment": g["treatment"].iloc[0], "slope": lr.slope})
    return pd.DataFrame(rows)


def compare_block_slopes(slopes_df: pd.DataFrame) -> pd.DataFrame:
    """Welch t-test of Stim vs Sham independent slopes, per block, Bonferroni
    across blocks. Reports t, df≈, p, p_bonf, Cohen's d, and group mean ± SD."""
    rows = []
    for block, g in slopes_df.groupby("block"):
        stim = g.loc[g["treatment"] == "Stim", "slope"].to_numpy(float)
        sham = g.loc[g["treatment"] == "Sham", "slope"].to_numpy(float)
        if len(stim) < 2 or len(sham) < 2:
            continue
        t, p = ttest_ind(stim, sham, equal_var=False)
        rows.append({
            "block": int(block), "n_stim": len(stim), "n_sham": len(sham),
            "stim_mean": stim.mean(), "stim_sd": stim.std(ddof=1),
            "sham_mean": sham.mean(), "sham_sd": sham.std(ddof=1),
            "t": t, "df": welch_df(stim, sham), "p": p, "cohens_d": cohens_d(stim, sham),
        })
    out = pd.DataFrame(rows).sort_values("block")
    out["p_bonf"] = multipletests(out["p"], alpha=config.ALPHA,
                                  method=config.MULTIPLICITY_METHOD)[1]
    out["sig_bonf"] = out["p_bonf"] < config.ALPHA
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Overall model comparison — random-effects structure via AIC/BIC              #
# --------------------------------------------------------------------------- #
# Nine specifications over the *whole* training period (all 8 blocks
# together, Block entered as a covariate rather than stratified on): three
# fixed-effects structures (no interaction; Treatment:Trial; Treatment:Block)
# crossed against random-effects structures (intercept only; + slope on
# Trial; + slope on Block) — a 3x3 grid:
#
#                        fixed effects            intercept | +Trial | +Block
#   no interaction       T+Trial+Block                M3    |   M7   |  M10
#   Treatment:Trial      +Treatment:Trial             M2    |   M1   |   M8
#   Treatment:Block      +Treatment:Block             M5    |  M11   |   M4
#
#   M1  y ~ treatment + trial_c + block_c + treatment:trial_c   (1 + trial_c | subject)
#   M2  y ~ treatment + trial_c + block_c + treatment:trial_c   (1 | subject)
#   M3  y ~ treatment + trial_c + block_c                       (1 | subject)
#   M4  y ~ treatment + trial_c + block_c + treatment:block_c   (1 + block_c | subject)
#   M5  y ~ treatment + trial_c + block_c + treatment:block_c   (1 | subject)
#   M7  y ~ treatment + trial_c + block_c                       (1 + trial_c | subject)
#   M8  y ~ treatment + trial_c + block_c + treatment:trial_c   (1 + block_c | subject)
#   M10 y ~ treatment + trial_c + block_c                       (1 + block_c | subject)
#   M11 y ~ treatment + trial_c + block_c + treatment:block_c   (1 + trial_c | subject)
#
# M1 vs M2 asks whether a random slope on Trial is warranted (fixed effects
# include Treatment:Trial); M3 vs M7 asks the same question with that
# interaction absent from the fixed effects. M4 vs M5 asks the analogous
# question for a random slope on Block (fixed effects include
# Treatment:Block); M1/M2 vs M8 asks whether a random Block slope beats a
# random Trial slope when the fixed interaction is Treatment:Trial instead;
# M10 is M3/M7's mirror (a random Block slope with no fixed interaction at
# all); M11 is M4/M5's mirror (a random Trial slope with the Treatment:Block
# fixed interaction). M3 vs M5 asks whether Treatment moderates the
# block-to-block trend (Treatment:Block) rather than the within-block trend
# (Treatment:Trial).
#
# All nine are fit by ML (not REML) — REML likelihoods aren't comparable once
# the fixed-effects structure changes (everything but M1/M2 drops or
# restructures terms relative to them), so ML keeps the AIC/BIC comparison
# valid across the whole ladder, at the cost of the fixed-effect SEs being the
# (slightly anti-conservative) ML ones rather than REML.
OVERALL_MODEL_SPECS = [
    {
        "model": "M1_intercept_and_trial_slope",
        "description": "Treatment + Trial + Block + Treatment:Trial, "
                       "random intercept + random slope on Trial by subject",
        "fixed": "treatment + trial_c + block_c + treatment:trial_c",
        "re_formula": "~trial_c",
    },
    {
        "model": "M2_intercept_only",
        "description": "Treatment + Trial + Block + Treatment:Trial, "
                       "random intercept only by subject",
        "fixed": "treatment + trial_c + block_c + treatment:trial_c",
        "re_formula": "~1",
    },
    {
        "model": "M3_no_interaction",
        "description": "Treatment + Trial + Block (no Treatment:Trial), "
                       "random intercept only by subject",
        "fixed": "treatment + trial_c + block_c",
        "re_formula": "~1",
    },
    {
        "model": "M4_intercept_and_block_slope",
        "description": "Treatment + Trial + Block + Treatment:Block, "
                       "random intercept + random slope on Block by subject",
        "fixed": "treatment + trial_c + block_c + treatment:block_c",
        "re_formula": "~block_c",
    },
    {
        "model": "M5_treatment_block_interaction",
        "description": "Treatment + Trial + Block + Treatment:Block, "
                       "random intercept only by subject",
        "fixed": "treatment + trial_c + block_c + treatment:block_c",
        "re_formula": "~1",
    },
    {
        "model": "M7_no_interaction_trial_slope",
        "description": "Treatment + Trial + Block (no Treatment:Trial), "
                       "random intercept + random slope on Trial by subject",
        "fixed": "treatment + trial_c + block_c",
        "re_formula": "~trial_c",
    },
    {
        "model": "M8_trial_interaction_block_slope",
        "description": "Treatment + Trial + Block + Treatment:Trial, "
                       "random intercept + random slope on Block by subject",
        "fixed": "treatment + trial_c + block_c + treatment:trial_c",
        "re_formula": "~block_c",
    },
    {
        "model": "M10_no_interaction_block_slope",
        "description": "Treatment + Trial + Block (no Treatment:Trial), "
                       "random intercept + random slope on Block by subject",
        "fixed": "treatment + trial_c + block_c",
        "re_formula": "~block_c",
    },
    {
        "model": "M11_block_interaction_trial_slope",
        "description": "Treatment + Trial + Block + Treatment:Block, "
                       "random intercept + random slope on Trial by subject",
        "fixed": "treatment + trial_c + block_c + treatment:block_c",
        "re_formula": "~trial_c",
    },
]


def _prep_overall_frame(trials_df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """One row per post-baseline run, all 8 blocks pooled.

    ``trial_c`` = trial-within-block (1..6, from `prep.add_block_labels`),
    centred. ``block_c`` = block number (1..8), centred: a single linear
    "improves by block" covariate, matching how "Trial" is a single
    coefficient rather than a per-block dummy. Centring is a
    reparameterisation only — it does not change AIC/BIC.
    """
    col = f"{measure}_improve"
    d = trials_df[trials_df["block"].fillna(0) > 0].copy()
    d = d.dropna(subset=[col, "trial_in_block", "block", "treatment"])
    d = d[d["treatment"].isin(["Stim", "Sham"])].copy()
    d["treatment"] = pd.Categorical(d["treatment"], categories=["Sham", "Stim"])

    trial = d["trial_in_block"].astype(float)
    d["trial_c"] = trial - trial.mean()

    block = d["block"].astype(float)
    d["block_c"] = block - block.mean()

    d["y"] = d[col]
    return d


def tidy_mixedlm_coefficients(fit, model: str = "") -> pd.DataFrame:
    """S2-style {beta, se, 95% CI, stat, p} table for any fitted MixedLM."""
    ci = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in fit.bse.index:  # variance components etc.
            continue
        beta, se = float(fit.params[term]), float(fit.bse[term])
        rows.append({
            "model": model, "term": term, "beta": beta, "se": se,
            "ci_low": float(ci.loc[term, 0]), "ci_high": float(ci.loc[term, 1]),
            "z": beta / se if se else np.nan,
            "p": float(fit.pvalues[term]) if term in fit.pvalues.index else np.nan,
        })
    return pd.DataFrame(rows)


# Every overall-model spec is fit with a single optimizer — powell — so every
# row in the comparison table comes from the same optimization method (lbfgs,
# the package default elsewhere via config.LMM_OPTIMIZER, failed to converge
# on the random-slope specs here; powell converges on every spec, every
# measure).
_OVERALL_OPTIMIZER = "powell"


def _fit_mixedlm_best(formula: str, data: pd.DataFrame, groups, re_formula: str):
    """Fit with `_OVERALL_OPTIMIZER`; return (fit, converged, optimizer_used).
    Raises if the fit itself errors out (e.g. a singular random-effects design)
    — the caller reports that as `converged=False` rather than silently
    swapping in a different optimizer or model family."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        fit = md.fit(reml=False, method='powell',
                     maxiter=config.LMM_MAXITER, disp=False)
    return fit, bool(getattr(fit, "converged", False)), 'powell'


def fit_overall_models(trials_df: pd.DataFrame,
                       measure: str = "RMSE_Needle") -> tuple[pd.DataFrame, dict]:
    """Fit every spec in `OVERALL_MODEL_SPECS`; return (comparison table,
    {model_name: fitted MixedLMResults}).

    Every model is fit with the same optimizer (`_OVERALL_OPTIMIZER`, currently
    "powell") so the AIC/BIC comparison isn't confounded by different specs
    landing on different optimizers. The comparison table has one row per
    model: formula, re_formula, optimizer, convergence, n_obs, n_groups,
    n_params, log-likelihood, AIC, BIC, and delta_aic/delta_bic relative to the
    best model in the set (lower is better for all four). No OLS fallback here
    — mixing model families would invalidate the AIC/BIC comparison the
    function exists to make; a model that fails to fit is reported with
    `converged=False` and NaN metrics instead of being silently swapped out
    for a different model class or optimizer.
    """
    d = _prep_overall_frame(trials_df, measure)
    rows, fits = [], {}

    for spec in OVERALL_MODEL_SPECS:
        formula = f"y ~ {spec['fixed']}"
        row = {
            "model": spec["model"], "description": spec["description"],
            "formula": formula, "re_formula": spec["re_formula"],
            "n_obs": len(d), "n_groups": d["subject"].nunique(),
        }
        try:
            fit, converged, optimizer = _fit_mixedlm_best(
                formula, d, d["subject"], spec["re_formula"])
            fits[spec["model"]] = fit
            row.update({
                "optimizer": optimizer, "converged": converged,
                "n_params": int(fit.df_modelwc),
                "log_likelihood": float(fit.llf),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
            })
        except Exception as exc:  # pragma: no cover
            fits[spec["model"]] = None
            row.update({"optimizer": None, "converged": False, "n_params": np.nan,
                        "log_likelihood": np.nan, "aic": np.nan, "bic": np.nan,
                        "error": str(exc)})
        rows.append(row)

    table = pd.DataFrame(rows)
    table["delta_aic"] = table["aic"] - table["aic"].min()
    table["delta_bic"] = table["bic"] - table["bic"].min()
    table["best_aic"] = table["aic"] == table["aic"].min()
    table["best_bic"] = table["bic"] == table["bic"].min()
    return table, fits


# --------------------------------------------------------------------------- #
# Baseline equivalence (Day 0)                                                 #
# --------------------------------------------------------------------------- #
def baseline_equivalence(trials_df: pd.DataFrame, measures=config.BEHAVIOR_MEASURES) -> pd.DataFrame:
    """Welch t-tests: do groups differ at baseline in mean level or in slope?"""
    base = trials_df[trials_df["day_num"] == config.BASELINE_DAY]
    rows = []
    for m in measures:
        lvl = base.groupby(["subject", "treatment"])[m].mean().reset_index()
        slp = []
        for (s, t), g in base.groupby(["subject", "treatment"]):
            gg = g.sort_values("trial")
            if len(gg) >= 2:
                slp.append({"subject": s, "treatment": t,
                            "slope": linregress(gg["run_num"], gg[m]).slope})
        slp = pd.DataFrame(slp)
        for label, frame, val in [("mean_level", lvl, m), ("slope", slp, "slope")]:
            a = frame.loc[frame["treatment"] == "Stim", val]
            b = frame.loc[frame["treatment"] == "Sham", val]
            t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            rows.append({"measure": m, "quantity": label, "t": t, "p": p,
                         "stim_mean": a.mean(), "sham_mean": b.mean()})
    return pd.DataFrame(rows)
