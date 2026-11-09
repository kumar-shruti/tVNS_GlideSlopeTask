"""
Smoke + checklist-regression tests. Run: `pytest -q` from the package root.

The behaviour tests need only the committed flight-performance CSV. Physio / EEG
tests skip when their inputs are absent.
"""
import pandas as pd
import pytest

from tvns_sensorimotor import influence, config, prep, stats_behavior as sb, stats_physio as sp, retention, cluster_permutation as cp, figures


@pytest.fixture(scope="module")
def trials():
    if not config.DATA.performance.exists():
        pytest.skip("performance CSV not present")
    return prep.build_trials_df()


def test_trials_shape(trials):
    assert trials["subject"].nunique() == 28
    assert set(trials["treatment"].unique()) == {"Stim", "Sham"}
    assert sorted(trials["block"].dropna().unique()) == list(range(9))


def test_B1_sign_convention(trials):
    # improvement = baseline - run  ->  block means predominantly positive
    post = trials[trials["block"] > 0]
    assert (post.groupby("block")["RMSE_Needle_improve"].mean() > 0).mean() >= 0.5


def test_S2_table1_columns(trials):
    t1 = sb.fit_block_models(trials, "RMSE_Needle")
    for col in ("beta", "se", "ci_low", "ci_high", "p", "p_bonf", "n_obs",
                "n_groups", "ols_fallback", "re_var_slope"):
        assert col in t1.columns
    assert t1["coef"].str.startswith("beta").any()


def test_S1_bonferroni_kills_block1(trials):
    t1 = sb.fit_block_models(trials, "RMSE_Needle")
    b1 = t1[(t1.block == 1) & (t1.term == "treatment[T.Stim]")].iloc[0]
    assert b1["p"] < 0.05          # uncorrected "significant"
    assert b1["p_bonf"] > 0.05     # gone after Bonferroni across 8 blocks


def test_B10_independent_slopes_no_treatment_leakage(trials):
    slopes = sb.independent_subject_slopes(trials, "RMSE_Needle")
    # every subject-block slope comes from its own 4-6 point OLS
    assert slopes["slope"].notna().all()
    comp = sb.compare_block_slopes(slopes)
    for col in ("t", "df", "p", "p_bonf", "cohens_d", "stim_sd", "sham_sd"):
        assert col in comp.columns


def test_B10_slopes_differ_from_blup_path(trials):
    """The corrected (independent) test should not reproduce the manuscript's
    'Block 1, 6, 8 significant after Bonferroni' claim."""
    comp = sb.compare_block_slopes(sb.independent_subject_slopes(trials, "RMSE_Needle"))
    assert not comp["sig_bonf"].any()


def test_interpret_slopes_direction():
    # synthetic: sham slope -0.002, stim slope -0.001 (shallower) via +0.001 interaction
    tbl = pd.DataFrame({
        "model": ["Pupil Diameter"] * 3,
        "term": ["trial_c", "treatment[T.Stim]", "trial_c:treatment[T.Stim]"],
        "beta": [-0.002, -0.2, 0.001], "p": [1e-7, 0.4, 0.6],
    })
    out = sp.interpret_slopes(tbl).iloc[0]
    assert "shallow" in out["verbal"]
    assert out["stim_slope_beta1_plus_beta3"] == pytest.approx(-0.001)


def test_overall_model_comparison(trials):
    table, fits = sb.fit_overall_models(trials, "RMSE_Needle")
    assert list(table["model"]) == [s["model"] for s in sb.OVERALL_MODEL_SPECS]
    for col in ("aic", "bic", "delta_aic", "delta_bic", "log_likelihood",
                "n_params", "converged", "optimizer"):
        assert col in table.columns
    # every spec fits under the powell optimizer
    # (lbfgs alone fails to converge on the random-slope models here)
    assert table["converged"].all()
    assert (table["optimizer"] == "powell").all()
    assert (table["delta_aic"] >= 0).all() and (table["delta_bic"] >= 0).all()
    assert (table["delta_aic"] == 0).sum() == 1

    best_model = table.loc[table["aic"].idxmin(), "model"]
    coefs = sb.tidy_mixedlm_coefficients(fits[best_model], best_model)
    assert {"term", "beta", "se", "ci_low", "ci_high", "p"} <= set(coefs.columns)


def test_figure1_pct_learning_curves(trials):
    import matplotlib
    matplotlib.use("Agg")
    fig = figures.figure1_pct_learning_curves(trials)
    assert len(fig.get_axes()) == len(config.BEHAVIOR_MEASURES)
    for ax in fig.get_axes():
        assert ax.get_ylabel() == "% improvement from baseline"

    # the Stim line's first plotted point should match an independent
    # (baseline - RMSE) / baseline * 100 computation for that trial
    d = trials[(trials["block"] > 0) & (trials["treatment"] == "Stim")]
    pct = (d["RMSE_Needle_baseline"] - d["RMSE_Needle"]) / d["RMSE_Needle_baseline"] * 100
    expected_first = pct.groupby(d["trial"]).mean().sort_index().iloc[0]

    ax = fig.get_axes()[0]  # RMSE_Needle panel; Stim plotted first -> lines[0]
    stim_line = ax.get_lines()[0]
    assert stim_line.get_ydata()[0] == pytest.approx(expected_first)


def test_pointwise_ttests(trials):
    table = cp.pointwise_ttests(trials)
    assert set(table["measure"]) == set(config.BEHAVIOR_MEASURES)
    assert set(table["value_kind"]) == {"improve", "pct_change"}
    for col in ("trial", "n_stim", "n_sham", "stim_mean", "sham_mean", "t", "df", "p", "cohens_d"):
        assert col in table.columns
    assert table["p"].between(0, 1).all()
    # 48 post-baseline trials per (measure, value_kind) combination
    assert (table.groupby(["measure", "value_kind"]).size() == 48).all()


def test_behavior_cluster_test(trials):
    # small n_permutations keeps this fast; structure matters more than power
    clusters = cp.run_cluster_tests(trials, n_permutations=200)
    assert set(clusters) == set(config.BEHAVIOR_MEASURES)
    for hits in clusters.values():
        for t0, t1, p in hits:
            assert t0 <= t1
            assert p < config.ALPHA

    sentence = cp.summarize_clusters(clusters)
    for m in config.BEHAVIOR_MEASURES:
        assert m in sentence
    if not any(clusters.values()):
        assert "no statistically significant cluster" in sentence


def test_behavior_cluster_figure_builds(trials):
    import matplotlib
    matplotlib.use("Agg")
    clusters = cp.run_cluster_tests(trials, n_permutations=200)
    fig = cp.figure_cluster_overlay(trials, clusters)
    assert len(fig.get_axes()) == len(config.BEHAVIOR_MEASURES)


def test_eeg_gamma_band_is_30_55():
    assert config.EEG_BANDS["gamma"] == (30.0, 55.0)
    assert config.EEG_BANDPASS_HZ[1] <= 60.0


@pytest.fixture(scope="module")
def retention_df():
    if not config.DATA.performance.exists():
        pytest.skip("performance CSV not present")
    return prep.build_retention_df()


def test_retention_df_shape(retention_df):
    # every day in the study is present: baseline, all 4 training days, both retention days
    assert set(retention_df["day"].unique()) == {"D00", "D01", "D02", "D03", "D04", "DR1", "DR2"}
    # D00 is 0% by construction
    d00 = retention_df[retention_df["day"] == "D00"]
    assert (d00["RMSE_Needle_pct_change"] == 0).all()
    assert (d00["RMSE_Airspeed_pct_change"] == 0).all()
    # day_order is monotone and continues past the training days into retention
    for day, order in [("D00", 0), ("D01", 1), ("D02", 2), ("D03", 3), ("D04", 4),
                       ("DR1", 5), ("DR2", 6)]:
        assert retention_df.loc[retention_df["day"] == day, "day_order"].eq(order).all()


def test_retention_pct_change_formula(retention_df):
    # positive = better than baseline (baseline - day), matching the
    # "positive = better" convention used by prep.add_baseline_correction
    r = retention_df[retention_df["day"] == "DR1"].iloc[0]
    expected = (r["RMSE_Needle_baseline"] - r["RMSE_Needle"]) / r["RMSE_Needle_baseline"] * 100
    assert r["RMSE_Needle_pct_change"] == pytest.approx(expected)
    # subjects improved (lower RMSE) by DR1 relative to D00, on average -> positive
    assert retention_df.loc[retention_df["day"] == "DR1", "RMSE_Needle_pct_change"].mean() > 0


def test_retention_ttest_tables(retention_df):
    raw = retention.retention_raw_ttests(retention_df)
    pct = retention.retention_pct_change_ttests(retention_df)
    for tbl, measures in [(raw, config.BEHAVIOR_MEASURES),
                          (pct, [f"{m}_pct_change" for m in config.BEHAVIOR_MEASURES])]:
        assert set(tbl["measure"]) <= set(measures)
        assert set(tbl["day"]) <= {"DR1", "DR2"}
        for col in ("t", "df", "p", "p_bonf", "cohens_d", "stim_sem", "sham_sem"):
            assert col in tbl.columns


def test_retention_figures_build(retention_df):
    import matplotlib
    matplotlib.use("Agg")
    raw = retention.retention_raw_ttests(retention_df)
    pct = retention.retention_pct_change_ttests(retention_df)
    for fig in (retention.figure_retention_raw(retention_df, raw),
               retention.figure_retention_pct_change(retention_df, pct),
               retention.figure_retention_trajectory(retention_df)):
        assert fig.get_axes()


def test_influence_leverage_sums_to_p(trials):
    fits = influence._fit_block_mixed(trials, "RMSE_Needle")
    lev = influence.subject_leverage(fits[1])
    assert len(lev) == trials["subject"].nunique()
    assert abs(lev.sum() - 4) < 0.05


def test_influence_loso_overall_small(trials):
    """LOSO on a subset of subjects (kept small for speed)."""
    keep = sorted(trials["subject"].unique())[::4]   # 7 subjects
    sub = trials[trials["subject"].isin(keep)]
    loso, subj, summ, full, model = influence.loso_overall_model(
        sub, "RMSE_Needle", model="M3_no_interaction")
    assert len(subj) == len(keep)
    assert (summ["n_refits"] == len(keep)).all()
    assert (summ["beta_loso_min"] <= summ["beta_full"] + 1e-9).all()
    assert (summ["beta_loso_max"] >= summ["beta_full"] - 1e-9).all()
    for col in ("dfbetas", "sign_change", "sig_change"):
        assert col in loso.columns


def test_full_three_way_model(trials):
    tbl, fit = sb.fit_full_model(trials, "RMSE_Needle")
    assert list(tbl["term"]) == list(sb.FULL_MODEL_TERMS)   # b0..b7
    assert tbl["converged"].all()
    assert fit.cov_re.shape == (2, 2)                       # u0 + u1*block
