"""
Cluster-based permutation testing (Stim vs Sham) for behavioural performance:
RMSE_Needle and RMSE_Airspeed across the training trial axis.

Same 1-D independent-samples cluster-permutation method already used for
peripheral physiology (`stats_physio.trial_cluster_test`) and EEG
(`eeg.run_cluster_tests`) — see `_common.cluster_test_over_index` for the
shared implementation — applied here to `trials_df`'s baseline-corrected
`{measure}_improve` columns (positive = better; B1), the same quantity Figure
1 plots, not raw RMSE.

Every model elsewhere in this package (Table 1's per-block slope, the overall
model-selection ladder) assumes a single linear-in-Trial or linear-in-Block
structure. A cluster permutation test makes no such shape assumption: it
finds any contiguous run of trials where the groups differ, correcting for
multiplicity by the cluster-mass null distribution rather than a Bonferroni
across many individual per-trial tests (which would be badly underpowered
here — order-N_trials comparisons on N=28 subjects).

An empty result is reported as exactly that — "no cluster survived" — never
silently dropped or explained away (cf. B7 / `eeg.summarize_clusters`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from ._common import cluster_test_over_index

STIM, SHAM = config.COLOR_STIM, config.COLOR_SHAM


# --------------------------------------------------------------------------- #
# Test                                                                         #
# --------------------------------------------------------------------------- #
def trial_cluster_test(trials_df: pd.DataFrame, measure: str,
                       n_permutations: int | None = None) -> list[tuple[float, float, float]]:
    """1-D cluster-based permutation test (Stim vs Sham) over the trial axis,
    on ``{measure}_improve`` (baseline-corrected, positive = better; B1).

    Post-baseline trials only (``block > 0``), matching Figure 1. Returns
    ``[(trial_start, trial_end, p), ...]`` for clusters with ``p < alpha``, or
    ``[]`` when nothing survives.
    """
    n_permutations = config.EEG_N_PERMUTATIONS if n_permutations is None else n_permutations
    d = trials_df[trials_df["block"] > 0]
    return cluster_test_over_index(
        d, index_col="trial", value_col=f"{measure}_improve",
        n_permutations=n_permutations, alpha=config.ALPHA)


def pointwise_ttests(trials_df: pd.DataFrame,
                     measures=config.BEHAVIOR_MEASURES,
                     value_kinds: tuple[str, ...] = ("improve", "pct_change")) -> pd.DataFrame:
    """Uncorrected Welch t-test, Stim vs Sham, at every individual trial —
    the raw per-trial statistic `trial_cluster_test`'s cluster-mass procedure
    thresholds and corrects, exposed here directly for inspection.

    ``value_kinds`` selects ``{measure}_improve`` (baseline - RMSE) and/or
    ``{measure}_pct_change`` (``improve / baseline * 100``) — same B1
    "positive = better" sign either way, just raw units vs. percent.

    At ~48 trials this is exactly the multiple-comparisons problem B7/S1
    warn about elsewhere in this package: a single trial's uncorrected
    ``p < alpha`` here is not a finding by itself (~2-3 of ~48 trials would
    cross .05 by chance alone) — use `trial_cluster_test` for the corrected
    version. Returns one tidy row per (measure, value_kind, trial).
    """
    from scipy.stats import ttest_ind

    from ._common import cohens_d, welch_df

    d = trials_df[trials_df["block"] > 0]
    rows = []
    for measure in measures:
        for kind in value_kinds:
            col = f"{measure}_{kind}"
            for trial, g in d.groupby("trial"):
                stim = g.loc[g["treatment"] == "Stim", col].dropna().to_numpy(float)
                sham = g.loc[g["treatment"] == "Sham", col].dropna().to_numpy(float)
                if len(stim) < 2 or len(sham) < 2:
                    continue
                t, p = ttest_ind(stim, sham, equal_var=False)
                rows.append({
                    "measure": measure, "value_kind": kind, "trial": trial,
                    "n_stim": len(stim), "n_sham": len(sham),
                    "stim_mean": stim.mean(), "sham_mean": sham.mean(),
                    "t": t, "df": welch_df(stim, sham), "p": p,
                    "cohens_d": cohens_d(stim, sham),
                })
    return (pd.DataFrame(rows)
           .sort_values(["measure", "value_kind", "trial"])
           .reset_index(drop=True))


def run_cluster_tests(trials_df: pd.DataFrame,
                      measures=config.BEHAVIOR_MEASURES,
                      n_permutations: int | None = None) -> dict[str, list]:
    """`trial_cluster_test` for every measure. Returns {measure: hits}."""
    return {m: trial_cluster_test(trials_df, m, n_permutations) for m in measures}


def summarize_clusters(clusters: dict[str, list]) -> str:
    """The sentence the Results section should contain per measure (B7)."""
    lines = []
    for measure, hits in clusters.items():
        if not hits:
            lines.append(f"{measure}: no statistically significant cluster "
                        "survived permutation correction.")
        else:
            spans = "; ".join(f"trials {t0:.0f}–{t1:.0f} (p={p:.3f})"
                              for t0, t1, p in hits)
            lines.append(f"{measure}: {len(hits)} cluster(s) survived — {spans}.")
    return " ".join(lines)


# --------------------------------------------------------------------------- #
# Figure — learning curves with cluster spans overlaid                        #
# --------------------------------------------------------------------------- #
def figure_cluster_overlay(trials_df: pd.DataFrame, clusters: dict[str, list],
                           measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    """Same view as `figures.figure1_learning_curves`, with a black bar over
    any trial range where a cluster survived (cf. `figures.figure3_physio`'s
    `clusters` overlay)."""
    from .figures import _block_boundaries

    d = trials_df[trials_df["block"] > 0]
    bounds, _, _ = _block_boundaries(trials_df)
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
        for (t0, t1, _p) in clusters.get(m, []):
            ax.plot([t0, t1], [ax.get_ylim()[1]] * 2, color="k", lw=3, clip_on=False)
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_xlabel("Trial")
        ax.set_ylabel("Improvement from baseline (baseline − RMSE)")
    axes[0].legend(frameon=False)
    fig.suptitle("Cluster-based permutation test — Stim vs Sham (black bar = surviving cluster)",
                weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(trials_df: pd.DataFrame | None = None, save: bool = True,
        measures=config.BEHAVIOR_MEASURES, n_permutations: int | None = None) -> dict:
    """Run the cluster test for every measure, print the honest summary
    sentence, build the overlay figure, and optionally save both to
    ``config.OUT_DIR`` (same location `run_all` uses)."""
    from . import prep

    cfg = config.RunConfig(save_figures=save)
    trials_df = prep.build_trials_df() if trials_df is None else trials_df

    clusters = run_cluster_tests(trials_df, measures, n_permutations)
    sentence = summarize_clusters(clusters)
    print(sentence)
    fig = figure_cluster_overlay(trials_df, clusters, measures)

    if save:
        rows = [{"measure": m, "trial_start": t0, "trial_end": t1, "p": p}
                for m, hits in clusters.items() for (t0, t1, p) in hits]
        table = pd.DataFrame(rows, columns=["measure", "trial_start", "trial_end", "p"])
        table.to_csv(cfg.out_dir / "behavior_clusters.csv", index=False)
        path = cfg.out_dir / f"figure_behavior_clusters.{config.SAVE_FORMAT}"
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
        print(f"  wrote {path}")

    return {"clusters": clusters, "sentence": sentence, "figure": fig}


if __name__ == "__main__":
    run()
