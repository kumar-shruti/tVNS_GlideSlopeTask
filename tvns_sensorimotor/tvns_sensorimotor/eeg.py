"""
EEG band power + cluster-permutation testing for the Figure-5 topographies and
trajectories.

Checklist items addressed here:
  B5  – gamma defaults to 30–55 Hz (config.EEG_BANDS), consistent with the
        1–60 Hz bandpass + 60/120 Hz notch actually used. The band and the
        filter are printed together in `describe_preprocessing()`.
  B7  – `run_cluster_tests` returns the number of surviving clusters (typically
        zero); `summarize_clusters` produces the exact sentence the Results
        should use. No thresholding trick is applied to manufacture a cluster.
  M1  – Welch window/overlap, permutation count, threshold, tail and the
        neighbour (adjacency) definition are all in config and echoed here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import config, io


def describe_preprocessing() -> str:
    lo, hi = config.EEG_BANDPASS_HZ
    n1, n2 = config.EEG_NOTCH_HZ
    g0, g1 = config.EEG_BANDS["gamma"]
    return (
        f"Continuous EEG band-pass {lo:g}–{hi:g} Hz, notches at {n1:g}/{n2:g} Hz; "
        f"band power by Welch PSD (nperseg = {config.EEG_WELCH_NPERSEG_SEC:g} s, "
        f"{config.EEG_WELCH_OVERLAP:.0%} overlap). Bands: "
        + ", ".join(f"{k} {v[0]:g}–{v[1]:g} Hz" for k, v in config.EEG_BANDS.items())
        + f". Gamma is {g0:g}–{g1:g} Hz because the {hi:g} Hz low-pass and "
        f"{n1:g} Hz notch remove everything above it (B5). "
        f"Cluster test: {config.EEG_N_PERMUTATIONS} permutations, "
        f"tail = {config.EEG_CLUSTER_TAIL}, threshold = {config.EEG_CLUSTER_THRESHOLD}, "
        f"channel adjacency from the {config.EEG_MONTAGE} montage."
    )


def bandpower(X: np.ndarray, fs: int, fmin: float, fmax: float) -> np.ndarray:
    """(n, ch, t) -> (n, ch) mean PSD in [fmin, fmax] via Welch."""
    from scipy.signal import welch

    nperseg = int(min(X.shape[-1], fs * config.EEG_WELCH_NPERSEG_SEC))
    noverlap = int(nperseg * config.EEG_WELCH_OVERLAP)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
    sel = (freqs >= fmin) & (freqs <= fmax)
    return psd[..., sel].mean(axis=-1)


def compute_band_power(raw: io.RawEEG | None = None) -> dict:
    raw = io.load_eeg_raw() if raw is None else raw
    return {name: bandpower(raw.X, config.EEG_FS, lo, hi)
            for name, (lo, hi) in config.EEG_BANDS.items()}, raw


def _mne_info_adjacency():
    import mne

    montage = mne.channels.make_standard_montage(config.EEG_MONTAGE)
    info = mne.create_info(montage.ch_names, sfreq=config.EEG_FS, ch_types="eeg")
    info.set_montage(montage)
    adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    return info, adjacency


@dataclass
class ClusterResult:
    band: str
    epoch: str
    n_stim: int
    n_sham: int
    n_sig_clusters: int
    min_cluster_p: float
    diff_topo: np.ndarray      # (ch,) Stim - Sham mean band power
    sig_mask: np.ndarray       # (ch,) bool


def run_cluster_tests(band_power: dict, raw: io.RawEEG) -> list[ClusterResult]:
    import mne
    from mne.stats import spatio_temporal_cluster_test

    info, adjacency = _mne_info_adjacency()
    block_idx = raw.blocks - raw.blocks.min()
    tx = raw.treatment
    sham_lab, stim_lab = sorted(np.unique(tx))[:2]

    results = []
    for band in config.EEG_BANDS:
        for epoch, blocks in config.EEG_EPOCHS.items():
            m = np.isin(block_idx, blocks)
            stim_d = band_power[band][m & (tx == stim_lab)]
            sham_d = band_power[band][m & (tx == sham_lab)]
            if len(stim_d) < 3 or len(sham_d) < 3:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, clusters, cluster_p, _ = spatio_temporal_cluster_test(
                    [stim_d[:, None, :], sham_d[:, None, :]],
                    n_permutations=config.EEG_N_PERMUTATIONS,
                    adjacency=adjacency, seed=42,
                    tail=config.EEG_CLUSTER_TAIL,
                    threshold=config.EEG_CLUSTER_THRESHOLD,
                    out_type="mask", n_jobs=1, verbose=False,
                )
            sig = np.zeros(len(info["ch_names"]), bool)
            for cl, p in zip(clusters, cluster_p):
                if p < config.ALPHA:
                    sig[np.asarray(cl).squeeze()] = True
            results.append(ClusterResult(
                band=band, epoch=epoch, n_stim=len(stim_d), n_sham=len(sham_d),
                n_sig_clusters=int((np.asarray(cluster_p) < config.ALPHA).sum()),
                min_cluster_p=float(np.min(cluster_p)) if len(cluster_p) else np.nan,
                diff_topo=stim_d.mean(0) - sham_d.mean(0), sig_mask=sig,
            ))
    return results


def summarize_clusters(results: list[ClusterResult]) -> str:
    """The sentence the Results section should contain (B7)."""
    total_sig = sum(r.n_sig_clusters for r in results)
    if total_sig == 0:
        best = min((r.min_cluster_p for r in results if np.isfinite(r.min_cluster_p)),
                   default=float("nan"))
        return ("Cluster-based permutation testing revealed no statistically "
                "significant spatiotemporal clusters for any band or training "
                f"epoch (all cluster p > {best:.2f}). Topographic differences in "
                "Figure 5 are descriptive only.")
    return (f"{total_sig} cluster(s) survived permutation correction: "
            + "; ".join(f"{r.band}/{r.epoch} p={r.min_cluster_p:.3f}"
                        for r in results if r.n_sig_clusters))


def block_trajectories(band_power: dict, raw: io.RawEEG) -> dict:
    """Per-band subject-mean ± SEM band power across the 8 blocks, by group."""
    block_idx = raw.blocks - raw.blocks.min()
    subj, tx = raw.subject, raw.treatment
    sham_lab, stim_lab = sorted(np.unique(tx))[:2]
    out = {}
    for band in config.EEG_BANDS:
        rec = {}
        for lab, key in [(stim_lab, "Stim"), (sham_lab, "Sham")]:
            subs = np.unique(subj[tx == lab])
            mat = np.full((len(subs), config.N_TRAINING_BLOCKS), np.nan)
            for i, s in enumerate(subs):
                for b in range(config.N_TRAINING_BLOCKS):
                    sel = (subj == s) & (block_idx == b)
                    if sel.any():
                        mat[i, b] = band_power[band][sel].mean()
            rec[key] = {
                "mean": np.nanmean(mat, 0),
                "sem": np.nanstd(mat, 0) / np.sqrt(np.sum(~np.isnan(mat), 0)),
            }
        out[band] = rec
    return out
