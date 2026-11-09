"""
Subject-level influence analysis (leave-one-subject-out, LOSO) for the
behavioural mixed models, plus coefficient ("beta") plots for those models.

Observations within a subject are not independent, so the unit removed is a
*subject*: each of the n subjects is dropped in turn, the model is refit on the
remaining n-1, and the change in the fixed effects is measured.

Per dropped subject i and fixed-effect coefficient k:
  DFBETAS_ik = (b_k - b_(i)k) / se_(i)k          cutoff |DFBETAS| > 2/sqrt(n)
  Cook's D_i = (b - b_(i))' V(b)^-1 (b - b_(i)) / p     cutoffs 4/n and 1
               (V(b) = full-data fixed-effect covariance, p = # fixed effects)
  sign change         sign(b_(i)k) != sign(b_k)
  significance change (p_(i)k < alpha) != (p_k < alpha)
  leverage_i = tr( V(b) X_i' V_i^-1 X_i )         cutoff 2p/n
               (generalised hat-matrix trace for subject i's block of the GLS
                fit; sums to p over subjects)

LOSO summary per coefficient: full-data beta, min/max over the n refits, number
of refits still significant, number of sign flips, the most influential subject.

Two model families are covered:
  * Table 1 per-block models (`stats_behavior.fit_block_models`), refit through
    the same function so the OLS fallback and the Bonferroni-across-blocks
    correction are reproduced exactly in every refit;
  * the pooled model of record, `stats_behavior.FULL_MODEL_SPEC`
    (treatment × trial × block, all interactions, random intercept + block
    slope by subject), fit by REML; any ladder spec can be passed instead.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config
from . import stats_behavior as sb

TABLE1_TERMS = ["trial_c", "treatment[T.Stim]", "trial_c:treatment[T.Stim]"]
TERM_LABELS = {
    "Intercept": "Intercept",
    "trial_c": "Trial (within-block slope)",
    "block_c": "Block",
    "treatment[T.Stim]": "Stim − Sham",
    "trial_c:treatment[T.Stim]": "Stim × Trial",
    "treatment[T.Stim]:trial_c": "Stim × Trial",
    "treatment[T.Stim]:block_c": "Stim × Block",
}
TABLE1_PANEL_TITLES = {
    "trial_c": "β1  within-block slope (Sham)",
    "treatment[T.Stim]": "β2  mean level, Stim − Sham",
    "trial_c:treatment[T.Stim]": "β3  slope, Stim − Sham",
}


def cutoffs(n_subjects: int, n_params: int) -> dict:
    return {"dfbetas": 2 / np.sqrt(n_subjects), "cooks_4n": 4 / n_subjects,
            "cooks_1": 1.0, "leverage": 2 * n_params / n_subjects}


# --------------------------------------------------------------------------- #
# Generic helpers                                                              #
# --------------------------------------------------------------------------- #
def subject_leverage(fit) -> pd.Series:
    """Generalised leverage of each subject in a fitted MixedLM:
    tr( (X'V^-1X)^-1 X_i' V_i^-1 X_i ). Sums to the number of fixed effects."""
    model = fit.model
    fe = fit.fe_params.index
    cov_fe = fit.cov_params().loc[fe, fe].to_numpy()
    G = np.asarray(fit.cov_re, float)
    out = {}
    for label, X, Z in zip(model.group_labels, model.exog_li, model.exog_re_li):
        V = Z @ G @ Z.T + fit.scale * np.eye(len(X))
        out[label] = float(np.trace(cov_fe @ X.T @ np.linalg.solve(V, X)))
    return pd.Series(out, name="leverage")


def _cooks(b_full: np.ndarray, b_loo: np.ndarray, cov_full: np.ndarray) -> float:
    d = b_full - b_loo
    return float(d @ np.linalg.solve(cov_full, d) / len(d))


def _add_flags(loso: pd.DataFrame, n_subjects: int, alpha: float) -> pd.DataFrame:
    loso["dfbetas"] = (loso["beta_full"] - loso["beta"]) / loso["se"]
    loso["dfbetas_flag"] = loso["dfbetas"].abs() > 2 / np.sqrt(n_subjects)
    loso["sign_change"] = np.sign(loso["beta"]) != np.sign(loso["beta_full"])
    loso["sig_change"] = (loso["p"] < alpha) != (loso["p_full"] < alpha)
    return loso


def _summarize(loso: pd.DataFrame, keys: list[str], alpha: float,
               bonf: bool = False) -> pd.DataFrame:
    rows = []
    for key, g in loso.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        top = g.loc[g["dfbetas"].abs().idxmax()]
        row = dict(zip(keys, key))
        row.update({
            "beta_full": g["beta_full"].iloc[0], "p_full": g["p_full"].iloc[0],
            "sig_full": g["p_full"].iloc[0] < alpha,
            "beta_loso_min": g["beta"].min(), "beta_loso_max": g["beta"].max(),
            "n_refits": len(g),
            "n_sig": int((g["p"] < alpha).sum()),
            "n_sign_change": int(g["sign_change"].sum()),
            "n_sig_change": int(g["sig_change"].sum()),
            "n_dfbetas_flag": int(g["dfbetas_flag"].sum()),
            "max_abs_dfbetas": abs(top["dfbetas"]),
            "most_influential_subject": top["dropped_subject"],
        })
        if bonf:
            row.update({"p_bonf_full": g["p_bonf_full"].iloc[0],
                        "sig_bonf_full": g["p_bonf_full"].iloc[0] < alpha,
                        "n_sig_bonf": int((g["p_bonf"] < alpha).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Table 1 — per-block models                                                   #
# --------------------------------------------------------------------------- #
def _fit_block_mixed(trials_df: pd.DataFrame, measure: str) -> dict:
    """Full-data per-block MixedLM fits (same spec as Table 1), for Cook's D
    covariance and leverage. Blocks that fall back to OLS are omitted."""
    d = sb._prep_block_frame(trials_df, measure)
    fits = {}
    for block, g in d.groupby("block"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("y ~ trial_c * treatment", g, groups=g["subject"],
                                  re_formula="~trial_c").fit(
                    reml=config.REML, method=config.LMM_OPTIMIZER,
                    maxiter=config.LMM_MAXITER, disp=False)
            if getattr(fit, "converged", True):
                fits[int(block)] = fit
        except Exception:  # pragma: no cover
            pass
    return fits


def loso_block_models(trials_df: pd.DataFrame, measure: str = "RMSE_Needle",
                      table1: pd.DataFrame | None = None):
    """LOSO over subjects for the Table 1 per-block models.

    Returns (loso, subjects, summary):
      loso     one row per (block, term, dropped subject)
      subjects one row per (block, subject): Cook's D, leverage, flags
      summary  one row per (block, term): LOSO range, # significant refits
               (raw and Bonferroni-across-blocks), sign/significance changes
    """
    alpha = config.ALPHA
    full = sb.fit_block_models(trials_df, measure) if table1 is None else table1
    full = full.set_index(["block", "term"])
    subjects = sorted(trials_df["subject"].unique())
    n = len(subjects)

    rows = []
    for s in subjects:
        t = sb.fit_block_models(trials_df[trials_df["subject"] != s], measure)
        t["dropped_subject"] = s
        rows.append(t)
    loso = pd.concat(rows, ignore_index=True)
    loso = loso.join(full[["beta", "se", "p", "p_bonf", "ols_fallback"]]
                     .add_suffix("_full"), on=["block", "term"])
    loso.insert(0, "measure", loso.pop("measure"))
    loso = _add_flags(loso, n, alpha)
    loso["sig_bonf_change"] = (loso["p_bonf"] < alpha) != (loso["p_bonf_full"] < alpha)

    # subject-level Cook's D (all 4 fixed effects jointly) + leverage
    fits = _fit_block_mixed(trials_df, measure)
    fe_terms = ["Intercept", "trial_c", "treatment[T.Stim]", "trial_c:treatment[T.Stim]"]
    subj_rows = []
    for block, fit in fits.items():
        cov = fit.cov_params().loc[fe_terms, fe_terms].to_numpy()
        b_full = fit.fe_params[fe_terms].to_numpy()
        lev = subject_leverage(fit)
        lb = loso[loso["block"] == block].pivot(index="dropped_subject",
                                               columns="term", values="beta")
        for s in lb.index:
            subj_rows.append({"measure": measure, "block": block, "subject": s,
                              "cooks_d": _cooks(b_full, lb.loc[s, fe_terms].to_numpy(), cov),
                              "leverage": lev.get(s, np.nan)})
    subj = pd.DataFrame(subj_rows)
    cut = cutoffs(n, len(fe_terms))
    subj["cooks_flag_4n"] = subj["cooks_d"] > cut["cooks_4n"]
    subj["cooks_flag_1"] = subj["cooks_d"] > cut["cooks_1"]
    subj["leverage_flag"] = subj["leverage"] > cut["leverage"]
    treat = subjects_treatment(trials_df)
    subj["treatment"] = subj["subject"].map(treat)

    summary = _summarize(loso[loso["term"] != "Intercept"],
                         ["measure", "block", "term"], alpha, bonf=True)
    summary.insert(3, "coef", summary["term"].map(sb.COEF_LABELS))
    summary["n_ols_fallback"] = (loso[loso["term"] != "Intercept"]
                                 .groupby(["block", "term"], sort=False)["ols_fallback"]
                                 .sum().to_numpy())
    return loso, subj, summary


def subjects_treatment(trials_df: pd.DataFrame) -> pd.Series:
    return trials_df.groupby("subject")["treatment"].first()


# --------------------------------------------------------------------------- #
# Pooled overall model                                                         #
# --------------------------------------------------------------------------- #
def loso_overall_model(trials_df: pd.DataFrame, measure: str = "RMSE_Needle",
                       model: str | dict | None = None, reml: bool = config.REML):
    """LOSO for one pooled model on the `_prep_overall_frame` data.

    `model` is a spec dict (default `stats_behavior.FULL_MODEL_SPEC`) or the
    name of a spec in `stats_behavior.OVERALL_MODEL_SPECS`. Returns
    (loso, subjects, summary, full_coefs, model_name)."""
    alpha = config.ALPHA
    d = sb._prep_overall_frame(trials_df, measure)
    if model is None:
        model = sb.FULL_MODEL_SPEC
    spec = model if isinstance(model, dict) else next(
        s for s in sb.OVERALL_MODEL_SPECS if s["model"] == model)
    model = spec["model"]
    formula = f"y ~ {spec['fixed']}"

    fit, converged = sb.fit_mixedlm(formula, d, spec["re_formula"], reml)
    fe = list(fit.fe_params.index)
    b_full = fit.fe_params.to_numpy()
    cov_full = fit.cov_params().loc[fe, fe].to_numpy()
    full = sb.tidy_mixedlm_coefficients(fit, model).set_index("term").loc[fe]
    lev = subject_leverage(fit)

    subjects = sorted(d["subject"].unique())
    n = len(subjects)
    rows, subj_rows = [], []
    for s in subjects:
        dd = d[d["subject"] != s]
        f, conv = sb.fit_mixedlm(formula, dd, spec["re_formula"], reml)
        for term in fe:
            ci = f.conf_int()
            rows.append({"measure": measure, "model": model, "term": term,
                         "dropped_subject": s, "beta": f.fe_params[term],
                         "se": f.bse[term], "p": f.pvalues[term],
                         "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                         "converged": conv,
                         "beta_full": full.loc[term, "beta"],
                         "se_full": full.loc[term, "se"], "p_full": full.loc[term, "p"]})
        subj_rows.append({"measure": measure, "model": model, "subject": s,
                          "cooks_d": _cooks(b_full, f.fe_params[fe].to_numpy(), cov_full),
                          "leverage": lev[s]})
    loso = _add_flags(pd.DataFrame(rows), n, alpha)
    subj = pd.DataFrame(subj_rows)
    cut = cutoffs(n, len(fe))
    subj["cooks_flag_4n"] = subj["cooks_d"] > cut["cooks_4n"]
    subj["cooks_flag_1"] = subj["cooks_d"] > cut["cooks_1"]
    subj["leverage_flag"] = subj["leverage"] > cut["leverage"]
    subj["treatment"] = subj["subject"].map(subjects_treatment(trials_df))
    summary = _summarize(loso, ["measure", "model", "term"], alpha)
    summary["n_not_converged"] = (~loso.groupby("term", sort=False)["converged"]
                                  .all()).astype(int).to_numpy()
    full = full.reset_index()
    full.insert(0, "measure", measure)
    full.insert(3, "label", full["term"].map(sb.FULL_MODEL_TERMS).fillna(full["term"]))
    full["converged"] = converged
    return loso, subj, summary, full, model


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
ACCENT = "#4c6ef5"     # LOSO range band


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)


def figure_block_betas(table1s: dict[str, pd.DataFrame],
                       summaries: dict[str, pd.DataFrame] | None = None) -> plt.Figure:
    """Table 1 betas per block: point = estimate, thin whisker = 95% CI,
    wide translucent bar = LOSO range (min–max over the n refits). Filled
    marker = p < .05 uncorrected; * above = Bonferroni-significant."""
    measures = list(table1s)
    fig, axes = plt.subplots(len(TABLE1_TERMS), len(measures),
                             figsize=(5.2 * len(measures), 2.6 * len(TABLE1_TERMS)),
                             sharex=True, squeeze=False)
    for j, m in enumerate(measures):
        t1 = table1s[m]
        summ = summaries.get(m) if summaries else None
        for i, term in enumerate(TABLE1_TERMS):
            ax = axes[i, j]
            _style(ax)
            r = t1[t1["term"] == term].sort_values("block")
            x = r["block"].to_numpy()
            if summ is not None:
                s = summ[summ["term"] == term].set_index("block").loc[x]
                ax.bar(x, s["beta_loso_max"] - s["beta_loso_min"],
                       bottom=s["beta_loso_min"], width=0.42, color=ACCENT,
                       alpha=0.22, lw=0, label="LOSO range", zorder=1)
            ax.errorbar(x, r["beta"], yerr=[r["beta"] - r["ci_low"], r["ci_high"] - r["beta"]],
                        fmt="none", ecolor=INK, elinewidth=1.2, capsize=0, zorder=2)
            sig = r["p"].to_numpy() < config.ALPHA
            ax.scatter(x[sig], r["beta"].to_numpy()[sig], s=42, color=INK, zorder=3,
                       label="p < .05")
            ax.scatter(x[~sig], r["beta"].to_numpy()[~sig], s=42, facecolor="white",
                       edgecolor=INK, lw=1.4, zorder=3, label="n.s.")
            for xx, yy, ok in zip(x, r["ci_high"], r["sig_bonf"]):
                if ok:
                    ax.text(xx, yy, "*", ha="center", va="bottom", color=INK, fontsize=13)
            ax.set_xticks(x)
            if i == 0:
                ax.set_title(m.replace("RMSE_", ""), weight="bold", color=INK)
            if j == 0:
                ax.set_ylabel(TABLE1_PANEL_TITLES[term], color=INK, fontsize=9)
            if i == len(TABLE1_TERMS) - 1:
                ax.set_xlabel("Training block")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=3, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Per-block LMM coefficients (improvement ~ trial × treatment)\n"
                 "95% CI whiskers · LOSO range bars · * Bonferroni p < .05",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def figure_fixed_effects(fulls: dict[str, pd.DataFrame],
                         summaries: dict[str, pd.DataFrame] | None = None) -> plt.Figure:
    """Dot-and-whisker plot of every fixed effect of the pooled model, one
    column per measure: dot = beta (filled if p < .05), whisker = 95% CI,
    translucent bar = LOSO range. The intercept sits in its own small panel
    because its scale would flatten the other terms."""
    measures = list(fulls)
    n_terms = len(next(iter(fulls.values())))
    fig, axes = plt.subplots(2, len(measures), squeeze=False,
                             figsize=(5.4 * len(measures), 0.55 * n_terms + 2.2),
                             gridspec_kw={"height_ratios": [1, n_terms - 1]})
    for j, m in enumerate(measures):
        f = fulls[m].reset_index(drop=True)
        s = summaries[m].set_index("term").loc[f["term"]] if summaries else None
        is_int = (f["term"] == "Intercept").to_numpy()
        for ax, mask in ((axes[0, j], is_int), (axes[1, j], ~is_int)):
            ff = f[mask].reset_index(drop=True)
            y = np.arange(len(ff))[::-1]
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color(MUTED)
            ax.spines["bottom"].set_color(MUTED)
            ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=0)
            if s is not None:
                ss = s[mask]
                ax.barh(y, ss["beta_loso_max"] - ss["beta_loso_min"],
                        left=ss["beta_loso_min"], height=0.45, color=ACCENT, alpha=0.22,
                        lw=0, label="LOSO range", zorder=1)
            ax.errorbar(ff["beta"].to_numpy(), y,
                        xerr=[(ff["beta"] - ff["ci_low"]).to_numpy(),
                              (ff["ci_high"] - ff["beta"]).to_numpy()],
                        fmt="none", ecolor=INK, elinewidth=1.3, zorder=2)
            sig = ff["p"].to_numpy() < config.ALPHA
            ax.scatter(ff["beta"][sig], y[sig], s=44, color=INK, zorder=3, label="p < .05")
            ax.scatter(ff["beta"][~sig], y[~sig], s=44, facecolor="white", edgecolor=INK,
                       lw=1.4, zorder=3, label="n.s.")
            for yy, (_, r) in zip(y, ff.iterrows()):
                ax.annotate(f"p={r['p']:.3f}" if r["p"] >= 0.001 else "p<.001",
                            (r["ci_high"], yy), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7, color=MUTED)
            labels = ff["label"] if "label" in ff else ff["term"]
            ax.set_yticks(y, labels, fontsize=8)
            ax.set_ylim(-0.6, len(ff) - 0.4)
            ax.margins(x=0.15)
        axes[0, j].set_title(m.replace("RMSE_", ""), weight="bold", color=INK)
        axes[1, j].set_xlabel("β (improvement from baseline, RMSE units)")
    h, l = axes[1, -1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=3, fontsize=8)
    model = next(iter(fulls.values()))["model"].iloc[0]
    fig.suptitle(f"Fixed effects — {model}\n"
                 "y ~ treatment × trial × block + (1 + block | subject), REML",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def figure_influence(block_subj: dict[str, pd.DataFrame],
                     overall_subj: dict[str, pd.DataFrame],
                     n_params_overall: dict[str, int]) -> plt.Figure:
    """Top row: Cook's D heatmap, subject × block, for the Table 1 models.
    Bottom row: Cook's D per subject (sorted) for the pooled model."""
    measures = list(block_subj)
    fig, axes = plt.subplots(2, len(measures), figsize=(5.6 * len(measures), 9.5),
                             squeeze=False, gridspec_kw={"height_ratios": [1.5, 1]})
    for j, m in enumerate(measures):
        bs = block_subj[m]
        n = bs["subject"].nunique()
        cut = cutoffs(n, 4)
        order = bs.groupby("subject")["treatment"].first().sort_values().index
        mat = bs.pivot(index="subject", columns="block", values="cooks_d").loc[order]
        ax = axes[0, j]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="Blues",
                       vmin=0, vmax=max(cut["cooks_4n"] * 2, np.nanmax(mat.to_numpy())))
        for (r, c), v in np.ndenumerate(mat.to_numpy()):
            if v > cut["cooks_4n"]:
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > cut["cooks_4n"] * 1.5 else INK)
        treat = bs.groupby("subject")["treatment"].first().loc[order]
        ax.set_yticks(range(len(order)),
                      [f"{s} ({t[:2]})" for s, t in zip(order, treat)], fontsize=6)
        ax.set_xticks(range(mat.shape[1]), mat.columns)
        ax.set_xlabel("Training block")
        ax.set_title(f"{m.replace('RMSE_', '')}: per-block Cook's D "
                     f"(labelled if > 4/n = {cut['cooks_4n']:.2f})", fontsize=9, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Cook's D")

        os_ = overall_subj[m].sort_values("cooks_d", ascending=False).reset_index(drop=True)
        cut_o = cutoffs(len(os_), n_params_overall[m])
        ax = axes[1, j]
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        x = np.arange(len(os_))
        ax.vlines(x, 0, os_["cooks_d"], color=GRID, lw=1.5, zorder=1)
        for grp, color in [("Stim", config.COLOR_STIM), ("Sham", config.COLOR_SHAM)]:
            g = os_[os_["treatment"] == grp]
            ax.scatter(g.index, g["cooks_d"], s=40, color=color, label=grp,
                       edgecolor="white", lw=1, zorder=3)
        ax.axhline(cut_o["cooks_4n"], color=MUTED, ls="--", lw=1)
        ax.text(len(os_) - 0.5, cut_o["cooks_4n"], f"4/n = {cut_o['cooks_4n']:.2f}",
                ha="right", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(x, os_["subject"], rotation=90, fontsize=6)
        ax.set_ylabel("Cook's D")
        ax.set_title(f"{m.replace('RMSE_', '')}: pooled {os_['model'].iloc[0].split('_')[0]} "
                     f"Cook's D per subject", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Subject-level influence (leave-one-subject-out)", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def figure_subject_trajectories(trials_df: pd.DataFrame, measure: str = "RMSE_Needle",
                                highlight: str = "P412") -> plt.Figure:
    """Raw `measure` vs trial-within-block, one small panel per training block,
    one thin line per subject (tinted by group), `highlight` drawn on top in
    ink so a subject flagged by the influence analysis can be eyeballed."""
    d = trials_df[trials_df["block"].fillna(0) > 0].dropna(subset=[measure])
    blocks = sorted(d["block"].astype(int).unique())
    fig, axes = plt.subplots(2, int(np.ceil(len(blocks) / 2)), sharex=True, sharey=True,
                             figsize=(12, 5.6), squeeze=False)
    colors = {"Stim": config.COLOR_STIM, "Sham": config.COLOR_SHAM}
    for ax, b in zip(axes.flat, blocks):
        g = d[d["block"] == b]
        for (subj, grp), s in g.groupby(["subject", "treatment"]):
            if subj == highlight:
                continue
            s = s.sort_values("trial_in_block")
            ax.plot(s["trial_in_block"], s[measure], color=colors.get(grp, MUTED),
                    lw=0.9, alpha=0.35, zorder=1)
        h = g[g["subject"] == highlight].sort_values("trial_in_block")
        ax.plot(h["trial_in_block"], h[measure], color=INK, lw=2.2, marker="o", ms=4,
                zorder=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if measure.endswith("_improve"):
            ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=0)
        ax.set_title(f"Block {b}", fontsize=9, color=INK)
        ax.set_xticks(range(1, 7))
    for ax in axes[-1]:
        ax.set_xlabel("Trial in block")
    ylabel = (f"{measure.removesuffix('_improve').replace('_', ' ')} improvement\n"
              "(baseline − RMSE; + = better)" if measure.endswith("_improve")
              else measure.replace("_", " "))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=9)
    grp = d.loc[d["subject"] == highlight, "treatment"].iloc[0]
    handles = [plt.Line2D([], [], color=config.COLOR_STIM, lw=1.5, alpha=0.6, label="Stim"),
               plt.Line2D([], [], color=config.COLOR_SHAM, lw=1.5, alpha=0.6, label="Sham"),
               plt.Line2D([], [], color=INK, lw=2.2, marker="o", ms=4,
                          label=f"{highlight} ({grp})")]
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"{measure} by trial within each training block — one line per subject",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run(trials_df: pd.DataFrame, measures=config.BEHAVIOR_MEASURES,
        table1s: dict | None = None) -> dict:
    out = {"block": {}, "overall": {}}
    for m in measures:
        t1 = table1s.get(m) if table1s else None
        if t1 is None:
            t1 = sb.fit_block_models(trials_df, m)
        print(f"[influence] LOSO Table 1 models — {m}")
        loso, subj, summ = loso_block_models(trials_df, m, table1=t1)
        out["block"][m] = {"table1": t1, "loso": loso, "subjects": subj, "summary": summ}
        print(f"[influence] LOSO pooled model {sb.FULL_MODEL_SPEC['model']} — {m}")
        loso_o, subj_o, summ_o, full_o, model = loso_overall_model(trials_df, m)
        out["overall"][m] = {"loso": loso_o, "subjects": subj_o, "summary": summ_o,
                             "full": full_o, "model": model}

    out["fig_block_betas"] = figure_block_betas(
        {m: v["table1"] for m, v in out["block"].items()},
        {m: v["summary"] for m, v in out["block"].items()})
    out["fig_fixed_effects"] = figure_fixed_effects(
        {m: v["full"] for m, v in out["overall"].items()},
        {m: v["summary"] for m, v in out["overall"].items()})
    out["fig_influence"] = figure_influence(
        {m: v["subjects"] for m, v in out["block"].items()},
        {m: v["subjects"] for m, v in out["overall"].items()},
        {m: len(v["full"]) for m, v in out["overall"].items()})
    return out


def tables(res: dict) -> dict[str, pd.DataFrame]:
    """Concatenate per-measure results into the CSVs run_all writes."""
    cat = lambda fam, k: pd.concat([v[k] for v in res[fam].values()], ignore_index=True)
    return {
        "influence_table1_summary": cat("block", "summary"),
        "influence_table1_subjects": cat("block", "subjects"),
        "influence_table1_loso": cat("block", "loso"),
        "influence_overall_summary": cat("overall", "summary"),
        "influence_overall_subjects": cat("overall", "subjects"),
        "influence_overall_loso": cat("overall", "loso"),
        "full_model_coefficients": cat("overall", "full"),
    }


if __name__ == "__main__":  # pragma: no cover
    from . import prep
    from .run_all import _save_fig, _save_table
    cfg = config.RunConfig()
    res = run(prep.build_trials_df())
    for name, df in tables(res).items():
        _save_table(df, name, cfg)
    _save_fig(res["fig_block_betas"], "figure_lme_block_betas", cfg)
    _save_fig(res["fig_fixed_effects"], "figure_lme_fixed_effects", cfg)
    _save_fig(res["fig_influence"], "figure_influence", cfg)
