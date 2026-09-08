"""Determine best/worst channel per cluster and get channel depth info.

Ports clus_channel_infor.m and getNP_chanDepthInfo.m.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unitassay.isolate.io.ks_loader import load_ks_dir
from unitassay.isolate.io.cluster_groups import read_cluster_groups


def _waveform_shape_similarity(wf_a: NDArray, wf_b: NDArray) -> float:
    """Cosine similarity between two same-length template waveforms.

    1.0 = identical shape (same polarity, any positive scaling); 0.0 =
    unrelated/orthogonal shapes; degenerate (all-zero) waveforms return 0.0.
    """
    norm_a = np.linalg.norm(wf_a)
    norm_b = np.linalg.norm(wf_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(wf_a, wf_b) / (norm_a * norm_b))


def clus_channel_info(
    ks_dir: str | Path,
    shape_sim_threshold: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Get best (max energy) and worst (min energy) channel per cluster.

    Ports clus_channel_infor.m. For each cluster, determines the channel
    with highest waveform energy (max_site) and lowest energy (min_site),
    weighted by PC feature coverage.

    EXPERIMENTAL (this dev copy only): the lowest-energy channel used to be
    picked purely by PTP energy, with no check that it isn't still capturing
    a decayed copy of the unit's own peak-channel waveform (spatial
    bleed-through on dense/small-pitch probes) rather than genuine
    background. Candidates whose template waveform shape is too similar
    (cosine similarity > shape_sim_threshold) to the peak-channel waveform
    are now excluded before picking the lowest-energy remainder.

    Parameters
    ----------
    ks_dir : path to Kilosort output directory
    shape_sim_threshold : max allowed cosine similarity, in [0, 1], between
        a candidate noise channel's template waveform and the unit's own
        peak-channel waveform. Above this, the candidate is treated as
        decayed unit signal rather than independent noise and skipped.

    Returns
    -------
    max_site : array of best channel indices (one per cluster)
    min_site : array of worst channel indices (one per cluster)
    noise_ch_flagged : bool array, True where no local channel satisfied
        both the PC-coverage floor AND the shape-dissimilarity requirement,
        so min_site fell back to the old PC-coverage-only choice (i.e. the
        returned noise channel may still be contaminated by decay — worth
        surfacing rather than silently accepting).
    """
    ks_dir = Path(ks_dir)

    sp = load_ks_dir(ks_dir, exclude_noise=False)
    pc_feat_ind = sp.pc_feat_ind
    spike_temps = sp.clu
    temps = sp.temps

    # Read cluster labels
    ks_file = ks_dir / "cluster_KSLabel.tsv"
    if not ks_file.exists():
        for name in ("cluster_group.tsv", "cluster_groups.csv"):
            candidate = ks_dir / name
            if candidate.exists():
                ks_file = candidate
                break

    cids, _ = read_cluster_groups(ks_file)

    alpha = 1.0
    min_pc_threshold = 0.1

    max_site_list: list[int] = []
    min_site_list: list[int] = []
    noise_ch_flagged_list: list[bool] = []

    for clu_id in cids:
        my_spikes = np.where(spike_temps == clu_id)[0]
        if len(my_spikes) == 0:
            max_site_list.append(0)
            min_site_list.append(0)
            noise_ch_flagged_list.append(False)
            continue

        # Template waveform for this cluster
        this_clu_temp = temps[clu_id]  # [nTimePoints x nChannels]
        tmp_ind = pc_feat_ind[clu_id]  # local channel indices (0-indexed)
        this_clu_wfs = this_clu_temp[:, tmp_ind]  # [nTimePoints x nLocalChans]
        ptp_energy = np.sum(this_clu_wfs ** 2, axis=0)

        # Non-zero PC ratios
        my_pc_feat = sp.pc_feat[my_spikes, :3, :] if sp.pc_feat is not None else None
        n_local = len(tmp_ind)
        non_zeros_pc_ratios = np.zeros(n_local)
        if my_pc_feat is not None:
            for ch in range(min(n_local, my_pc_feat.shape[2])):
                pc_ch = my_pc_feat[:, :, ch]
                zero_mask = np.all(pc_ch == 0, axis=1)
                non_zeros_pc_ratios[ch] = 1.0 - np.sum(zero_mask) / pc_ch.shape[0]

        # Normalize
        max_energy = np.max(ptp_energy) if np.max(ptp_energy) > 0 else 1.0
        ptp_norm = ptp_energy / max_energy
        max_nzr = np.max(non_zeros_pc_ratios) if np.max(non_zeros_pc_ratios) > 0 else 1.0
        nzr_norm = non_zeros_pc_ratios / max_nzr

        combined_score = alpha * ptp_norm + (1 - alpha) * nzr_norm
        best_idx = int(np.argmax(combined_score))

        # Check if best channel has sufficient PC coverage
        if non_zeros_pc_ratios[best_idx] < 0.5:
            eligible = np.where(non_zeros_pc_ratios >= 0.5)[0]
            if len(eligible) > 0:
                best_idx = eligible[int(np.argmax(combined_score[eligible]))]

        max_site_list.append(int(tmp_ind[best_idx]))

        # Shape similarity of every local channel's waveform to the unit's
        # own peak-channel waveform — a candidate "noise" channel that's
        # still a decayed copy of the same shape is spatial bleed-through,
        # not independent background.
        peak_wf = this_clu_wfs[:, best_idx]
        shape_sim = np.array(
            [_waveform_shape_similarity(this_clu_wfs[:, ch], peak_wf) for ch in range(n_local)]
        )
        decay_like = shape_sim > shape_sim_threshold

        # Worst channel: lowest energy, excluding decay-like candidates first
        worst_idx = int(np.argmin(ptp_energy))
        flagged = False
        if non_zeros_pc_ratios[worst_idx] < min_pc_threshold or decay_like[worst_idx]:
            eligible = np.where(
                (non_zeros_pc_ratios >= min_pc_threshold) & (ptp_energy > 0) & (~decay_like)
            )[0]
            if len(eligible) > 0:
                worst_idx = eligible[int(np.argmin(ptp_energy[eligible]))]
            else:
                # No channel is both PC-covered and shape-dissimilar enough —
                # fall back to the old PC-coverage-only choice and flag it,
                # rather than silently returning a possibly-contaminated
                # noise channel as if it were clean.
                eligible_old = np.where(
                    (non_zeros_pc_ratios >= min_pc_threshold) & (ptp_energy > 0)
                )[0]
                if len(eligible_old) > 0:
                    worst_idx = eligible_old[int(np.argmin(ptp_energy[eligible_old]))]
                flagged = True

        min_site_list.append(int(tmp_ind[worst_idx]))
        noise_ch_flagged_list.append(flagged)

    return (
        np.array(max_site_list, dtype=np.int64),
        np.array(min_site_list, dtype=np.int64),
        np.array(noise_ch_flagged_list, dtype=np.bool_),
    )
