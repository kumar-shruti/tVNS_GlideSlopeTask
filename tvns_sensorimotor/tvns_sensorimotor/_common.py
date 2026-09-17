"""
Small stats/plot helpers shared across modules (kept private — not part of the
public API, just de-duplication so `stats_behavior` and `retention` don't each
redefine Cohen's d / Welch df / significance stars, and `stats_physio` and
`cluster_permutation` don't each redefine the 1-D cluster-permutation test).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp else np.nan


def welch_df(a, b) -> float:
    """Welch–Satterthwaite approximate degrees of freedom."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v1, v2 = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (v1 + v2) ** 2 / (v1 ** 2 / (len(a) - 1) + v2 ** 2 / (len(b) - 1))


def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def cluster_test_over_index(
    df: pd.DataFrame, index_col: str, value_col: str, *,
    group_col: str = "treatment", groups: tuple[str, str] = ("Stim", "Sham"),
    subject_col: str = "subject", n_permutations: int = 5000,
    alpha: float = 0.05, seed: int = 42,
) -> list[tuple[float, float, float]]:
    """1-D independent-samples cluster-based permutation test (`groups[0]` vs
    `groups[1]`) of `value_col` over `index_col`, one point per subject per
    index value (averaged first if there are repeats).

    Returns ``[(index_start, index_end, p), ...]`` for clusters with
    ``p < alpha``, or ``[]`` when nothing survives — callers must report that
    honestly (cf. `eeg.summarize_clusters`), never explain it away.
    """
    from mne.stats import permutation_cluster_test

    d = df.dropna(subset=[value_col, index_col, group_col])
    wide = d.pivot_table(index=[subject_col, group_col], columns=index_col,
                         values=value_col, aggfunc="mean")
    idx = wide.columns.to_numpy()
    a = wide.xs(groups[0], level=group_col).to_numpy()
    b = wide.xs(groups[1], level=group_col).to_numpy()
    # cluster test needs no NaNs; keep only index values present for both groups
    ok = ~np.isnan(a).any(0) & ~np.isnan(b).any(0)
    a, b, idx = a[:, ok], b[:, ok], idx[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, clusters, pvals, _ = permutation_cluster_test(
            [a, b], n_permutations=n_permutations, tail=0, seed=seed, verbose=False)
    hits = []
    for cl, p in zip(clusters, pvals):
        if p < alpha:
            ci = np.where(cl)[0]
            hits.append((float(idx[ci.min()]), float(idx[ci.max()]), float(p)))
    return hits
