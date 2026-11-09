"""
Regenerate every table and figure in one call.

    python -m tvns_sensorimotor.run_all              # behaviour only if that's
                                                     # all the data present
    python -m tvns_sensorimotor.run_all --all        # + physio + EEG + arousal

Each stage is independent and skipped (with a printed reason) when its inputs
are missing, so the behaviour pipeline always runs from the committed CSV.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from . import (config, prep, stats_behavior, stats_physio, cluster_permutation,
              eeg, arousal, retention, figures, influence)


def _save_fig(fig, name, cfg):
    if cfg.save_figures:
        path = cfg.out_dir / f"{name}.{config.SAVE_FORMAT}"
        fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
        print(f"  wrote {path}")


def _save_table(df: pd.DataFrame, name, cfg):
    path = cfg.out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def run(save: bool = True, do_physio: bool = True, do_eeg: bool = True,
        do_arousal: bool = True, do_retention: bool = True, do_cluster: bool = True,
        do_influence: bool = True) -> dict:
    cfg = config.RunConfig(save_figures=save)
    out: dict = {}

    # ---- Behaviour (always) --------------------------------------------------
    print("[behaviour] building trials_df …")
    trials = prep.build_trials_df()
    out["trials_df"] = trials

    out["baseline_equivalence"] = stats_behavior.baseline_equivalence(trials)
    out["table1"] = stats_behavior.fit_block_models(trials, "RMSE_Needle")
    out["table1_airspeed"] = stats_behavior.fit_block_models(trials, "RMSE_Airspeed")
    slopes = stats_behavior.independent_subject_slopes(trials, "RMSE_Needle")
    out["table3_slopes"] = stats_behavior.compare_block_slopes(slopes)

    if save:
        for k in ("baseline_equivalence", "table1", "table1_airspeed", "table3_slopes"):
            _save_table(out[k], k, cfg)
    _save_fig(figures.figure1_learning_curves(trials), "figure1_learning_curves", cfg)
    _save_fig(figures.figure1_pct_learning_curves(trials), "figure1_pct_learning_curves", cfg)
    _save_fig(figures.figure2_block_learning(trials, out["table1"]), "figure2_block_learning", cfg)

    # ---- Behaviour cluster permutation (Stim vs Sham) — only needs the same CSV #
    if do_cluster:
        clusters = cluster_permutation.run_cluster_tests(trials)
        out["behavior_clusters"] = clusters
        out["behavior_cluster_sentence"] = cluster_permutation.summarize_clusters(clusters)
        print("[cluster]", out["behavior_cluster_sentence"])
        if save:
            rows = [{"measure": m, "trial_start": t0, "trial_end": t1, "p": p}
                    for m, hits in clusters.items() for (t0, t1, p) in hits]
            _save_table(pd.DataFrame(rows, columns=["measure", "trial_start", "trial_end", "p"]),
                       "behavior_clusters", cfg)
        _save_fig(cluster_permutation.figure_cluster_overlay(trials, clusters),
                 "figure_behavior_clusters", cfg)

    # ---- Influence (LOSO) on the behavioural LMMs + beta plots ------------- #
    if do_influence:
        print("[influence] leave-one-subject-out refits …")
        inf = influence.run(trials, table1s={"RMSE_Needle": out["table1"],
                                             "RMSE_Airspeed": out["table1_airspeed"]})
        out["influence"] = inf
        if save:
            for name, df in influence.tables(inf).items():
                _save_table(df, name, cfg)
        _save_fig(inf["fig_block_betas"], "figure_lme_block_betas", cfg)
        _save_fig(inf["fig_fixed_effects"], "figure_lme_fixed_effects", cfg)
        _save_fig(inf["fig_influence"], "figure_influence", cfg)

    # ---- Retention (post-training DR days) — only needs the same CSV -------- #
    if do_retention:
        print("[retention] building retention_df …")
        try:
            ret = retention.run(save=save)
            out["retention_df"] = ret["retention_df"]
            out["retention_raw_ttests"] = ret["raw_ttests"]
            out["retention_pct_change_ttests"] = ret["pct_change_ttests"]
        except FileNotFoundError as exc:
            print(f"[retention] skipped — {exc}")

    # ---- Physiology --------------------------------------------------------- #
    physio_df = None
    if do_physio:
        try:
            physio_df = prep.build_physio_df(trials)
            out["physio_df"] = physio_df
            out["table2"] = stats_physio.fit_physio_models(physio_df)
            out["table2_interpretation"] = stats_physio.interpret_slopes(out["table2"])
            out["physio_missingness"] = prep.missingness_table(
                physio_df, config.PHYSIO_MEASURES)
            clusters = {}
            for m in config.PHYSIO_MEASURES:
                try:
                    clusters[m] = stats_physio.trial_cluster_test(physio_df, m)
                except Exception as exc:  # pragma: no cover
                    print(f"  [physio] cluster test {m} failed: {exc}")
            out["physio_clusters"] = clusters
            if save:
                for k in ("table2", "table2_interpretation", "physio_missingness"):
                    _save_table(out[k], k, cfg)
            _save_fig(figures.figure3_physio(physio_df, out["table2"],
                                             clusters=clusters), "figure3_physio", cfg)
        except FileNotFoundError as exc:
            print(f"[physio] skipped — {exc}")

    # ---- Arousal (Figure 4) ------------------------------------------------- #
    if do_arousal:
        try:
            adf = prep.build_arousal_df(physio_df)
            scores, loadings, evr = arousal.fit_pca(adf)
            out["arousal_loadings"] = loadings
            out["arousal_evr"] = evr
            out["arousal_models"] = arousal.pc_learning_models(arousal.add_phase(scores))
            if save:
                _save_table(loadings.reset_index(), "arousal_loadings", cfg)
                _save_table(out["arousal_models"], "arousal_models", cfg)
            _save_fig(figures.figure4_arousal(scores, loadings), "figure4_arousal", cfg)
        except FileNotFoundError as exc:
            print(f"[arousal] skipped — {exc}")
        except Exception:  # pragma: no cover
            print("[arousal] failed:\n" + traceback.format_exc())

    # ---- EEG (Figure 5) --------------------------------------------------- #
    if do_eeg:
        try:
            print("[eeg]", eeg.describe_preprocessing())
            band_power, raw = eeg.compute_band_power()
            results = eeg.run_cluster_tests(band_power, raw)
            out["eeg_clusters"] = results
            out["eeg_cluster_sentence"] = eeg.summarize_clusters(results)
            print("[eeg]", out["eeg_cluster_sentence"])
            out["eeg_trajectories"] = eeg.block_trajectories(band_power, raw)
            if save:
                _save_table(pd.DataFrame([vars(r) for r in results]).drop(
                    columns=["diff_topo", "sig_mask"]), "eeg_clusters", cfg)
        except FileNotFoundError as exc:
            print(f"[eeg] skipped — {exc}")
        except Exception:  # pragma: no cover
            print("[eeg] failed:\n" + traceback.format_exc())

    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run physio + EEG + arousal too")
    ap.add_argument("--no-retention", action="store_true",
                    help="skip the post-training DR-day retention analysis")
    ap.add_argument("--no-cluster", action="store_true",
                    help="skip the behaviour cluster-permutation test")
    ap.add_argument("--no-influence", action="store_true",
                    help="skip the leave-one-subject-out influence analysis (~3-4 min)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    run(save=not a.no_save, do_physio=a.all, do_eeg=a.all, do_arousal=a.all,
        do_retention=not a.no_retention, do_cluster=not a.no_cluster,
        do_influence=not a.no_influence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
