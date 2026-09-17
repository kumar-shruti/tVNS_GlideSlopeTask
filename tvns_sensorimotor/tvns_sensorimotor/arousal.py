"""
Figure 4: a physiology-derived arousal latent variable and its relationship to
learning.

PCA on within-subject z-scored [heart_rate, rmssd, frontocentral_theta,
posterior_alpha, pupil]; the leading component is the arousal axis. We then test
whether the PC-vs-improvement relationship differs by treatment and by training
phase (early vs late).

This section needs the derived EEG-feature and physio tables merged onto the
behaviour frame (see prep.build_arousal_df). It is exploratory-supporting, not a
primary confirmatory result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import config

FEATURES = ["heart_rate", "rmssd", "frontocentral_theta", "posterior_alpha", "pupil"]
PHASE_EARLY_N = 12
PHASE_LATE_N = 12


def fit_pca(df: pd.DataFrame, features=FEATURES, n_components: int = 3):
    """Within-subject standardise, then PCA. Returns (scores_df, loadings, evr)."""
    from sklearn.decomposition import PCA

    d = df.dropna(subset=features).copy()
    d[features] = d.groupby("subject")[features].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    d = d.dropna(subset=features)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(d[features].to_numpy())
    cols = [f"PC{i+1}" for i in range(n_components)]
    for i, c in enumerate(cols):
        d[c] = scores[:, i]
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=cols)
    return d, loadings, pca.explained_variance_ratio_


def add_phase(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lo, hi = d["trial"].min(), d["trial"].max()
    early = set(range(int(lo), int(lo) + PHASE_EARLY_N))
    late = set(range(int(hi) - PHASE_LATE_N + 1, int(hi) + 1))
    d = d[d["trial"].isin(early | late)].copy()
    d["phase"] = np.where(d["trial"].isin(early), "Early", "Late")
    return d


def pc_learning_models(df_phase: pd.DataFrame, pcs=("PC1", "PC2", "PC3")) -> pd.DataFrame:
    """Per PC: improvement ~ pc * treatment * phase + (1 | subject).
    Reports the pc:treatment:phase term (does arousal-coupling of learning shift
    with stimulation across training?)."""
    rows = []
    for pc in pcs:
        d = df_phase.dropna(subset=[pc, "RMSE_Needle_improve"]).copy()
        d["pc"] = d[pc]
        try:
            fit = smf.mixedlm("RMSE_Needle_improve ~ pc * treatment * phase",
                              d, groups=d["subject"]).fit(reml=False, disp=False)
            for term in fit.params.index:
                if term in ("Intercept", "Group Var"):
                    continue
                rows.append({"pc": pc, "term": term,
                             "beta": float(fit.params[term]),
                             "p": float(fit.pvalues[term])})
        except Exception as exc:  # pragma: no cover
            rows.append({"pc": pc, "term": "FIT_FAILED", "beta": np.nan,
                         "p": np.nan, "error": str(exc)})
    return pd.DataFrame(rows)
