"""
Post-training retention (DR-day) analysis: RMSE_Needle and RMSE_Airspeed,
Stim vs Sham, at the DR1/DR2 follow-up sessions.

Three views, each a table + a figure:

  1. Raw scores at each retention day.
     `retention_raw_ttests` (Welch t-test) + `figure_retention_raw` (bar,
     mean ± SEM).
  2. Baseline-normalized % improvement:
     ``(D00 baseline − day) / D00 baseline * 100`` — positive = lower RMSE
     than baseline (better), matching the "positive = better" convention used
     everywhere else in the package (`prep.add_baseline_correction`'s
     ``_improve`` columns).
     `retention_pct_change_ttests` + `figure_retention_pct_change`.
  3. The full % improvement trajectory, D00 (0 %, by construction) through
     every training day (D01..D04) and every retention day.
     `figure_retention_trajectory`.

This is the study's retention test (checklist S11 — the original notebook
dropped DR1/DR2 entirely and the manuscript says "no retention test
conducted"). Needs only the committed flight-performance CSV.

Usage
-----
>>> from tvns_sensorimotor import retention
>>> res = retention.run()                 # tables + figures, saved to outputs/
or just the pieces:
>>> from tvns_sensorimotor import prep, retention
>>> rdf = prep.build_retention_df()
>>> retention.retention_raw_ttests(rdf)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import config
from ._common import cohens_d, p_to_stars, welch_df

STIM, SHAM = config.COLOR_STIM, config.COLOR_SHAM


# --------------------------------------------------------------------------- #
# Stats                                                                        #
# --------------------------------------------------------------------------- #
def _welch_table(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Welch t-test Stim vs Sham per (value column, day); Bonferroni across all
    rows produced (S1-style — every extra comparison here is still corrected)."""
    rows = []
    for col in value_cols:
        for day, g in df.groupby("day"):
            stim = g.loc[g["treatment"] == "Stim", col].dropna().to_numpy(float)
            sham = g.loc[g["treatment"] == "Sham", col].dropna().to_numpy(float)
            if len(stim) < 2 or len(sham) < 2:
                continue
            t, p = ttest_ind(stim, sham, equal_var=False)
            rows.append({
                "measure": col, "day": day,
                "n_stim": len(stim), "n_sham": len(sham),
                "stim_mean": stim.mean(), "stim_sem": stim.std(ddof=1) / np.sqrt(len(stim)),
                "sham_mean": sham.mean(), "sham_sem": sham.std(ddof=1) / np.sqrt(len(sham)),
                "t": t, "df": welch_df(stim, sham), "p": p,
                "cohens_d": cohens_d(stim, sham),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonf"] = multipletests(out["p"], alpha=config.ALPHA,
                                      method=config.MULTIPLICITY_METHOD)[1]
        out["sig_bonf"] = out["p_bonf"] < config.ALPHA
    return out


def retention_raw_ttests(retention_df: pd.DataFrame,
                         measures=config.BEHAVIOR_MEASURES) -> pd.DataFrame:
    """Table: raw RMSE at each retention day, Stim vs Sham, Welch t-test."""
    d = retention_df[retention_df["day"].str.startswith(config.FOLLOWUP_DAY_PREFIX)]
    return _welch_table(d, list(measures))


def retention_pct_change_ttests(retention_df: pd.DataFrame,
                                measures=config.BEHAVIOR_MEASURES) -> pd.DataFrame:
    """Table: % improvement from baseline at each retention day, Stim vs Sham
    (positive = better than baseline; see module docstring for the sign)."""
    d = retention_df[retention_df["day"].str.startswith(config.FOLLOWUP_DAY_PREFIX)]
    return _welch_table(d, [f"{m}_pct_change" for m in measures])


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #
def _grouped_bar(ax, table: pd.DataFrame):
    days = table["day"].tolist()
    x = np.arange(len(days))
    w = 0.32
    ax.bar(x - w / 2, table["stim_mean"], w, yerr=table["stim_sem"],
           capsize=4, color=STIM, label="Stim")
    ax.bar(x + w / 2, table["sham_mean"], w, yerr=table["sham_sem"],
           capsize=4, color=SHAM, label="Sham")
    ax.set_xticks(x)
    ax.set_xticklabels(days)

    stars = table["p_bonf"].map(p_to_stars) if "p_bonf" in table else ["ns"] * len(days)
    for xi, (s, row) in enumerate(zip(stars, table.itertuples())):
        if s != "ns":
            y = max(row.stim_mean + row.stim_sem, row.sham_mean + row.sham_sem)
            ax.text(xi, y, s, ha="center", va="bottom", fontsize=13)


def figure_retention_raw(retention_df: pd.DataFrame, ttest_table: pd.DataFrame,
                         measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    """Bar chart: mean raw RMSE ± SEM, Stim vs Sham, per retention day."""
    fig, axes = plt.subplots(1, len(measures), figsize=(5.5 * len(measures), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, measures):
        t = ttest_table[ttest_table["measure"] == m].sort_values("day")
        _grouped_bar(ax, t)
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_ylabel("RMSE (mean ± SEM)")
        ax.set_xlabel("Retention day")
    axes[0].legend(frameon=False)
    fig.suptitle("Post-training retention — raw performance (* Bonferroni p < .05)",
                weight="bold")
    fig.tight_layout()
    return fig


def figure_retention_pct_change(retention_df: pd.DataFrame, ttest_table: pd.DataFrame,
                                measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    """Bar chart: mean % improvement from baseline ± SEM, Stim vs Sham, per
    retention day (positive = better than baseline)."""
    fig, axes = plt.subplots(1, len(measures), figsize=(5.5 * len(measures), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, measures):
        t = ttest_table[ttest_table["measure"] == f"{m}_pct_change"].sort_values("day")
        _grouped_bar(ax, t)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_ylabel("% improvement from baseline (mean ± SEM)")
        ax.set_xlabel("Retention day")
    axes[0].legend(frameon=False)
    fig.suptitle("Post-training retention — % improvement from baseline (* Bonferroni p < .05)",
                weight="bold")
    fig.tight_layout()
    return fig


def figure_retention_trajectory(retention_df: pd.DataFrame,
                                measures=config.BEHAVIOR_MEASURES) -> plt.Figure:
    """Line plot: % improvement from baseline, D00 (0 %) -> D01..D04 -> every
    retention day (positive = better than baseline)."""
    fig, axes = plt.subplots(1, len(measures), figsize=(5.5 * len(measures), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, measures):
        col = f"{m}_pct_change"
        d = retention_df.dropna(subset=[col])
        days = (d[["day", "day_order"]].drop_duplicates().sort_values("day_order"))
        for grp, color in [("Stim", STIM), ("Sham", SHAM)]:
            g = (d[d["treatment"] == grp].groupby(["day_order", "day"])[col]
                 .agg(["mean", "sem"]).reset_index().sort_values("day_order"))
            ax.errorbar(g["day_order"], g["mean"], yerr=g["sem"], marker="o",
                       capsize=4, color=color, lw=2, label=grp)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xticks(days["day_order"])
        ax.set_xticklabels(days["day"])
        ax.set_title(m.replace("RMSE_", ""))
        ax.set_xlabel("Day")
        ax.set_ylabel("% improvement from baseline (mean ± SEM)")
    axes[0].legend(frameon=False)
    fig.suptitle("Baseline-normalized trajectory: D00 → training days → retention days",
                weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(save: bool = True, measures=config.BEHAVIOR_MEASURES) -> dict:
    """Build retention_df, run all three analyses, optionally save to
    ``config.OUT_DIR`` (same location `run_all` uses)."""
    from . import prep

    cfg = config.RunConfig(save_figures=save)
    retention_df = prep.build_retention_df(measures)

    raw_tt = retention_raw_ttests(retention_df, measures)
    pct_tt = retention_pct_change_ttests(retention_df, measures)

    fig_raw = figure_retention_raw(retention_df, raw_tt, measures)
    fig_pct = figure_retention_pct_change(retention_df, pct_tt, measures)
    fig_traj = figure_retention_trajectory(retention_df, measures)

    if save:
        retention_df.to_csv(cfg.out_dir / "retention_df.csv", index=False)
        raw_tt.to_csv(cfg.out_dir / "retention_raw_ttests.csv", index=False)
        pct_tt.to_csv(cfg.out_dir / "retention_pct_change_ttests.csv", index=False)
        for fig, name in [(fig_raw, "figure_retention_raw"),
                          (fig_pct, "figure_retention_pct_change"),
                          (fig_traj, "figure_retention_trajectory")]:
            path = cfg.out_dir / f"{name}.{config.SAVE_FORMAT}"
            fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
            print(f"  wrote {path}")

    return {
        "retention_df": retention_df,
        "raw_ttests": raw_tt,
        "pct_change_ttests": pct_tt,
        "fig_raw": fig_raw,
        "fig_pct_change": fig_pct,
        "fig_trajectory": fig_traj,
    }


if __name__ == "__main__":
    run()
