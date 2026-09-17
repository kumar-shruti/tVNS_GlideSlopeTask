"""
The four manuscript figures. Each function takes prepared frames / fitted
tables and returns a matplotlib Figure; saving is the caller's job (run_all).

Exploratory plots from the original notebook (waterfall, baseline-vs-delta,
power-law, GAM/RF, per-subject correlation heatmaps, 3-D PCA) are NOT here —
they live in :mod:`tvns_sensorimotor.exploratory`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .stats_behavior import independent_subject_slopes, compare_block_slopes

STIM, SHAM = config.COLOR_STIM, config.COLOR_SHAM


def _block_boundaries(df: pd.DataFrame):
    b = (df[df["block"] > 0].groupby("block")["trial"]
         .agg(start="min", end="max").reset_index().sort_values("block"))
    return b["start"].to_numpy()[1:], (0.5 * (b["start"] + b["end"])).to_numpy(), \
        b["block"].astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# Figure 1 — learning curves                                                   #
# --------------------------------------------------------------------------- #
def figure1_learning_curves(trials_df: pd.DataFrame,
                            measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    d = trials_df[trials_df["block"] > 0]
    bounds, centers, labels = _block_boundaries(trials_df)
    fig, axes = plt.subplots(1, len(measures), figsize=(6 * len(measures), 4.2),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, measures):
        col = f"{m}_improve"
        for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
            g = d[d["treatment"] == grp].groupby("trial")[col]
            mean, sem = g.mean(), g.sem()
            ax.plot(mean.index, mean.values, color=color, label=grp, lw=2)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
        for x in bounds:
            ax.axvline(x, color="gray", ls=":", lw=1)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_xlabel("Trial")
        ax.set_ylabel("Improvement from baseline (baseline − RMSE)")
    axes[0].legend(frameon=False)
    fig.suptitle("Behavioural learning across training", weight="bold")
    fig.tight_layout()
    return fig


def figure1_pct_learning_curves(trials_df: pd.DataFrame,
                                measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    """Same view as `figure1_learning_curves`, in % improvement from baseline
    instead of raw units: ``(baseline - RMSE) / baseline * 100`` — positive =
    better than baseline, the same convention as
    `retention.build_retention_df`'s ``_pct_change`` columns (there per day;
    here per trial)."""
    d = trials_df[trials_df["block"] > 0]
    bounds, centers, labels = _block_boundaries(trials_df)
    fig, axes = plt.subplots(1, len(measures), figsize=(6 * len(measures), 4.2),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, measures):
        col = f"{m}_pct_change"
        for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
            g = d[d["treatment"] == grp].groupby("trial")[col]
            mean, sem = g.mean(), g.sem()
            ax.plot(mean.index, mean.values, color=color, label=grp, lw=2)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)
        for x in bounds:
            ax.axvline(x, color="gray", ls=":", lw=1)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_xlabel("Trial")
        ax.set_ylabel("% improvement from baseline")
    axes[0].legend(frameon=False)
    fig.suptitle("Behavioural learning across training (% improvement from baseline)",
                weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 — piecewise block fits + within-block slope distributions           #
# --------------------------------------------------------------------------- #
def figure2_block_learning(trials_df: pd.DataFrame,
                           block_table: pd.DataFrame,
                           measure: str = "RMSE_Needle") -> plt.Figure:
    col = f"{measure}_improve"
    d = trials_df[trials_df["block"] > 0].dropna(subset=[col, "trial_in_block"])
    bounds, _, _ = _block_boundaries(trials_df)

    slopes = independent_subject_slopes(trials_df, measure)
    slope_tests = compare_block_slopes(slopes).set_index("block")

    fig, (ax_fit, ax_vio) = plt.subplots(2, 1, figsize=(11, 8))

    # (A) trial-mean scatter + per-block OLS fit lines per group
    for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
        gm = d[d["treatment"] == grp].groupby("trial")[col].mean()
        ax_fit.scatter(gm.index, gm.values, s=16, color=color, alpha=0.8, label=grp)
    for block, g in d.groupby("block"):
        for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
            gg = g[g["treatment"] == grp]
            if gg["trial_in_block"].nunique() < 2:
                continue
            coef = np.polyfit(gg["trial"], gg[col], 1)
            xs = np.sort(gg["trial"].unique())
            ax_fit.plot(xs, np.polyval(coef, xs), color=color, lw=2)
    for x in bounds:
        ax_fit.axvline(x, color="gray", ls=":", lw=1)
    ax_fit.axhline(0, color="k", ls="--", lw=1)
    ax_fit.set_xlabel("Trial (global)")
    ax_fit.set_ylabel("Improvement from baseline")
    ax_fit.set_title(f"{measure.replace('RMSE_', '')} — per-block linear fits")
    ax_fit.legend(frameon=False)

    # (B) independent within-block subject slopes, Stim vs Sham, with Bonferroni stars
    blocks = sorted(slopes["block"].unique())
    off = 0.18
    for b in blocks:
        for grp, color, dx in [("Sham", SHAM, -off), ("Stim", STIM, +off)]:
            vals = slopes.loc[(slopes.block == b) & (slopes.treatment == grp), "slope"]
            if len(vals) == 0:
                continue
            parts = ax_vio.violinplot([vals.values], positions=[b + dx], widths=0.3,
                                      showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.55)
                body.set_edgecolor("black")
            ax_vio.scatter(np.full(len(vals), b + dx), vals, s=12, color="k", alpha=0.4)
        if b in slope_tests.index and slope_tests.loc[b, "sig_bonf"]:
            y = slopes.loc[slopes.block == b, "slope"].max()
            ax_vio.text(b, y, "*", ha="center", va="bottom", fontsize=15)
    ax_vio.axhline(0, color="k", ls="--", lw=1)
    ax_vio.set_xticks(blocks)
    ax_vio.set_xlabel("Block")
    ax_vio.set_ylabel("Within-block slope\n(improvement per run, independent OLS)")
    ax_vio.set_title("Subject-specific within-block learning rates "
                     "(* Bonferroni p < .05)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 — peripheral physiology                                             #
# --------------------------------------------------------------------------- #
def figure3_physio(physio_df: pd.DataFrame,
                   physio_table: pd.DataFrame,
                   measures=config.PHYSIO_MEASURES,
                   clusters: dict | None = None) -> plt.Figure:
    d = physio_df[physio_df["block"] > 0]
    bounds, _, _ = _block_boundaries(physio_df)
    clusters = clusters or {}

    fig, axes = plt.subplots(2, len(measures), figsize=(5 * len(measures), 8),
                             sharex="col")
    for j, m in enumerate(measures):
        ax_obs, ax_lmm = axes[0, j], axes[1, j]
        for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
            g = d[d["treatment"] == grp].groupby("trial")[m]
            mean, sem = g.mean(), g.sem()
            ax_obs.plot(mean.index, mean.values, color=color, lw=2, label=grp)
            ax_obs.fill_between(mean.index, mean - sem, mean + sem, color=color, alpha=0.2)

        row = physio_table[(physio_table["model"] == m)].set_index("term")
        if {"trial_c", "treatment[T.Stim]", "trial_c:treatment[T.Stim]"} <= set(row.index):
            b1 = row.loc["trial_c", "beta"]
            b2 = row.loc["treatment[T.Stim]", "beta"]
            b3 = row.loc["trial_c:treatment[T.Stim]", "beta"]
            tc = np.linspace(d["trial"].min(), d["trial"].max(), 100)
            tcc = tc - d["trial"].mean()
            b0 = d[m].mean()
            ax_lmm.plot(tc, b0 + b1 * tcc, color=SHAM, lw=2)
            ax_lmm.plot(tc, b0 + b2 + (b1 + b3) * tcc, color=STIM, lw=2)

        for ax in (ax_obs, ax_lmm):
            for x in bounds:
                ax.axvline(x, color="gray", ls=":", lw=1)
        for (x0, x1, _p) in clusters.get(m, []):
            ax_obs.plot([x0, x1], [ax_obs.get_ylim()[1]] * 2, color="k", lw=3, clip_on=False)
        ax_obs.set_title(m)
        ax_lmm.set_xlabel("Trial")
        ax_obs.set_ylabel("Observed mean ± SEM")
        ax_lmm.set_ylabel("LMM fit")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Peripheral physiology across training", weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 — arousal latent variable                                          #
# --------------------------------------------------------------------------- #
def figure4_arousal(scores_df: pd.DataFrame, loadings: pd.DataFrame,
                    pcs=("PC1", "PC2", "PC3")) -> plt.Figure:
    from .arousal import add_phase

    dph = add_phase(scores_df)
    fig, axes = plt.subplots(1, len(pcs) + 1, figsize=(5 * (len(pcs) + 1), 4.5))

    im = axes[0].imshow(loadings.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(loadings.shape[1]))
    axes[0].set_xticklabels(loadings.columns)
    axes[0].set_yticks(range(loadings.shape[0]))
    axes[0].set_yticklabels(loadings.index)
    axes[0].set_title("PCA loadings")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    for ax, pc in zip(axes[1:], pcs):
        for (grp, phase), sub in dph.groupby(["treatment", "phase"]):
            color = STIM if grp == "Stim" else SHAM
            ax.scatter(sub[pc], sub["RMSE_Needle_improve"], s=10, alpha=0.4,
                       color=color, marker="o" if phase == "Early" else "^",
                       label=f"{grp} {phase}")
        ax.set_xlabel(pc)
        ax.set_ylabel("Improvement")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Physiology-derived arousal vs learning", weight="bold")
    fig.tight_layout()
    return fig
