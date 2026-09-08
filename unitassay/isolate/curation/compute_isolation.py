"""Compute PC features and isolation scores per cluster.

Ports compute_isolation.m. For each cluster, extracts PC features on the
max-amplitude channel, segments the recording into time windows, and computes
isolation metrics (signal-to-noise score) against nearby noise channels.
Also finds neighboring clusters on the same channel and saves their PC features.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from unitassay.isolate.config import MindSnagConfig
from unitassay.isolate.io.ks_loader import load_ks_dir
from unitassay.isolate.io.cluster_groups import read_cluster_groups
from unitassay.isolate.io.hdf5_writer import write_sort_data_h5
from unitassay.isolate.utils.channel_info import clus_channel_info

logger = logging.getLogger(__name__)


def compute_isolation(
    cfg: MindSnagConfig,
    ks_dir: str | Path,
    output_dir: str | Path,
    clu_ids: list[int] | None = None,
) -> None:
    """Compute isolation scores for all (or specific) clusters.

    Parameters
    ----------
    cfg : pipeline configuration
    ks_dir : path to Kilosort output directory
    output_dir : path for pipeline output
    clu_ids : optional list of specific cluster IDs (0-indexed)
    """
    ks_dir = Path(ks_dir)
    output_dir = Path(output_dir)
    iso_win = cfg.isolation.window_sec

    # Remove old SortData files so stale results don't persist on re-run
    for old_file in output_dir.glob("*.SortData.h5"):
        old_file.unlink()
    logger.debug("Cleared old SortData files in %s", output_dir)

    sp = load_ks_dir(ks_dir, exclude_noise=False)
    max_site, min_site, noise_ch_flagged = clus_channel_info(
        ks_dir, shape_sim_threshold=cfg.isolation.noise_shape_sim_threshold
    )
    if noise_ch_flagged.any():
        logger.info(
            "%d/%d clusters have no noise channel that is both PC-covered and "
            "shape-dissimilar from the peak channel; falling back to the old "
            "PC-coverage-only choice for those (possible decay bleed-through).",
            int(noise_ch_flagged.sum()), len(noise_ch_flagged),
        )

    ks_file = ks_dir / "cluster_KSLabel.tsv"
    cids, cgs = read_cluster_groups(ks_file)
    good_clus_0 = cids[cgs == 2]  # 0-indexed good cluster IDs

    pc_feat_ind = sp.pc_feat_ind
    temps = sp.temps

    # Load NPclu for spike data
    npclu_file = output_dir / "NPclu.h5"
    if not npclu_file.exists():
        logger.error("NPclu.h5 not found in %s. Run extract_spikes first.", output_dir)
        return

    with h5py.File(npclu_file, "r") as f:
        npclu = np.array(f["NPclu"])  # [2 x N]
        rec_spike_times = npclu[0, :]
        rec_spike_temps = npclu[1, :].astype(int)  # 1-indexed
        temp_scaling = np.array(f["tempScalingAmps"]).flatten()
        pc_feat = np.array(f["pcFeat"])  # [nLocal x 3 x N] in MATLAB layout

    # Transpose pcFeat back to [N x 3 x nLocal] for Python indexing
    pc_feat = pc_feat.transpose(2, 1, 0)

    # Determine which clusters to process
    if clu_ids is not None:
        my_clus = [c + 1 for c in clu_ids]  # to 1-indexed
    else:
        my_clus = (cids + 1).tolist()

    for i_unit, clu in enumerate(my_clus):
        my_spikes = np.where(rec_spike_temps == clu)[0]

        frames: list[dict] = []

        if len(my_spikes) > 0 and max_site is not None and i_unit < len(max_site):
            spike_ch_ind = max_site[i_unit]

            # Template waveform
            if clu - 1 < temps.shape[0] and spike_ch_ind < temps.shape[2]:
                my_spike_wf = temps[clu - 1, :, spike_ch_ind].flatten()
            else:
                my_spike_wf = np.full(61, np.nan)

            # PC features for this cluster's channel
            my_amp = np.tile(
                temp_scaling[my_spikes].reshape(-1, 1, 1),
                (1, 3, 1),
            )
            if pc_feat_ind is not None and clu - 1 < len(pc_feat_ind):
                tmp_ind = pc_feat_ind[clu - 1]
                pc_ch_idx = np.where(tmp_ind == spike_ch_ind)[0]
                if len(pc_ch_idx) > 0:
                    my_pc_feat = pc_feat[my_spikes, :3, pc_ch_idx[0]:pc_ch_idx[0]+1] * my_amp
                else:
                    my_pc_feat = np.zeros((len(my_spikes), 3, 1))
            else:
                my_pc_feat = np.zeros((len(my_spikes), 3, 1))

            my_spike_times = rec_spike_times[my_spikes]

            # --- Find other clusters on the same channel ---
            other_unit_indices = np.where(max_site == max_site[i_unit])[0]
            other_unit_indices = other_unit_indices[other_unit_indices != i_unit]
            other_clu_ids = np.array([my_clus[idx] for idx in other_unit_indices])
            n_other = len(other_clu_ids)

            # Extract PC features and waveforms for neighbor clusters
            other_pc_feats = []
            other_spike_times_list = []
            other_spike_wfs = []
            for i_o, o_idx in enumerate(other_unit_indices):
                o_clu = my_clus[o_idx]
                o_spikes = np.where(rec_spike_temps == o_clu)[0]
                o_ch_ind = max_site[o_idx]

                # Other cluster waveform
                if o_clu - 1 < temps.shape[0] and o_ch_ind < temps.shape[2]:
                    other_spike_wfs.append(temps[o_clu - 1, :, o_ch_ind].flatten())
                else:
                    other_spike_wfs.append(np.full(61, np.nan))

                # Other cluster PC features
                if len(o_spikes) > 0 and pc_feat_ind is not None and o_clu - 1 < len(pc_feat_ind):
                    o_tmp_ind = pc_feat_ind[o_clu - 1]
                    o_pc_ch_idx = np.where(o_tmp_ind == o_ch_ind)[0]
                    o_amp = np.tile(
                        temp_scaling[o_spikes].reshape(-1, 1, 1),
                        (1, 3, 1),
                    )
                    if len(o_pc_ch_idx) > 0:
                        o_pc = pc_feat[o_spikes, :3, o_pc_ch_idx[0]:o_pc_ch_idx[0]+1] * o_amp
                    else:
                        o_pc = np.zeros((len(o_spikes), 3, 1))
                else:
                    o_pc = np.zeros((0, 3, 1))

                other_pc_feats.append(o_pc)
                other_spike_times_list.append(rec_spike_times[o_spikes])

            # Good/bad labels for neighbor clusters
            good_bad = np.zeros(n_other, dtype=int)
            if n_other > 0:
                good_bad[np.isin(other_clu_ids, good_clus_0 + 1)] = 1

            # Noise channel
            noise_ch_ind = min_site[i_unit] if i_unit < len(min_site) else spike_ch_ind
            if clu - 1 < temps.shape[0] and noise_ch_ind < temps.shape[2]:
                noise_wf = temps[clu - 1, :, noise_ch_ind].flatten()
            else:
                noise_wf = np.full(61, np.nan)

            if pc_feat_ind is not None and clu - 1 < len(pc_feat_ind):
                tmp_ind = pc_feat_ind[clu - 1]
                noise_ch_idx = np.where(tmp_ind == noise_ch_ind)[0]
                if len(noise_ch_idx) > 0:
                    my_pc_noise = pc_feat[my_spikes, :3, noise_ch_idx[0]:noise_ch_idx[0]+1] * my_amp
                else:
                    my_pc_noise = np.zeros_like(my_pc_feat)
            else:
                my_pc_noise = np.zeros_like(my_pc_feat)

            # Segment into time windows
            if len(rec_spike_times) > 0:
                frac_end = int(np.ceil(np.max(rec_spike_times) / iso_win)) * iso_win
                frac = np.arange(0, frac_end + iso_win, iso_win)
                n_frames = len(frac) - 1
            else:
                n_frames = 0

            if n_frames > 0:
                for i_frame in range(n_frames):
                    ind = np.where(
                        (my_spike_times >= frac[i_frame]) &
                        (my_spike_times <= frac[i_frame + 1])
                    )[0]

                    frame: dict = {
                        "clu_wf": my_spike_wf,
                        "noise_wf": noise_wf,
                        "unit_iso": 0,
                        "clu": int(clu),
                    }

                    if len(ind) > 0:
                        unit_feat = my_pc_feat[ind].squeeze(-1)
                        noise_feat = my_pc_noise[ind].squeeze(-1)
                        frame["unit"] = unit_feat
                        frame["noise"] = noise_feat
                        frame["mean_spike_amp"] = np.mean(unit_feat, axis=0)
                        frame["mean_noise_amp"] = np.mean(noise_feat, axis=0)
                        frame["sd_noise_amp"] = np.std(noise_feat, axis=0, ddof=0)
                        sd = frame["sd_noise_amp"]
                        if sd[0] > 0:
                            frame["score"] = float(
                                np.abs(frame["mean_spike_amp"][0] - frame["mean_noise_amp"][0]) / sd[0]
                            )
                        else:
                            frame["score"] = 0.0
                    else:
                        frame["unit"] = np.empty((0, 3))
                        frame["noise"] = np.empty((0, 3))

                    # Neighbor clusters
                    if n_other > 0:
                        other_feats_frame = []
                        for i_o in range(n_other):
                            o_times = other_spike_times_list[i_o]
                            o_ind = np.where(
                                (o_times >= frac[i_frame]) & (o_times <= frac[i_frame + 1])
                            )[0]
                            if len(o_ind) > 0:
                                other_feats_frame.append(other_pc_feats[i_o][o_ind].squeeze(-1))
                            else:
                                other_feats_frame.append(np.empty((0, 3)))
                        frame["other"] = other_feats_frame
                        frame["other_clu"] = other_clu_ids.astype(np.float64)
                        frame["other_good_bad"] = good_bad.astype(np.float64)
                        if other_spike_wfs:
                            frame["other_clu_wf"] = np.array(other_spike_wfs)
                    else:
                        frame["other"] = []
                        frame["other_clu"] = np.array([], dtype=np.float64)
                        frame["other_good_bad"] = np.array([], dtype=np.float64)

                    frames.append(frame)
            else:
                frames.append({
                    "clu_wf": my_spike_wf,
                    "noise_wf": noise_wf,
                    "unit": my_pc_feat.squeeze(-1),
                    "noise": my_pc_noise.squeeze(-1),
                    "unit_iso": 0,
                    "clu": int(clu),
                    "other": [],
                    "other_clu": np.array([], dtype=np.float64),
                })
        else:
            frames.append({
                "unit_iso": 0,
                "clu": int(clu),
            })

        # Save
        out_file = output_dir / f"{clu}.SortData.h5"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        write_sort_data_h5(out_file, frames)
        logger.info("Saved: %s", out_file)
