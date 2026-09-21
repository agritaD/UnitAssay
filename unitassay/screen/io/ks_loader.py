"""Load Kilosort output directory into a KilosortOutput dataclass.

Ports loadKSdir.m. Uses np.load() instead of readNPY (one line replaces
readNPY.m + readNPYheader.m).

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from unitassay.screen.io.cluster_groups import read_cluster_groups
from unitassay.screen.io.params_py_loader import load_params_py
from unitassay.screen.types import KilosortOutput


def load_ks_dir(
    ks_dir: str | Path,
    exclude_noise: bool = True,
    load_pcs: bool = True,
) -> KilosortOutput:
    """Load all Kilosort output from a directory.

    Parameters
    ----------
    ks_dir : path to Kilosort output directory
    exclude_noise : if True, remove spikes assigned to noise clusters
    load_pcs : if True, load PC features

    Returns
    -------
    KilosortOutput dataclass with all spike sorting data.
    """
    ks_dir = Path(ks_dir)

    # Load params
    params = load_params_py(ks_dir / "params.py")
    sample_rate = float(params.get("sample_rate", 30000))

    # Spike times
    ss = np.load(ks_dir / "spike_times.npy").flatten()
    st = ss.astype(np.float64) / sample_rate

    # Template assignments (0-indexed)
    spike_templates = np.load(ks_dir / "spike_templates.npy").flatten().astype(np.int64)

    # Cluster assignments
    spike_clusters_path = ks_dir / "spike_clusters.npy"
    if spike_clusters_path.exists():
        clu = np.load(spike_clusters_path).flatten().astype(np.int64)
    else:
        clu = spike_templates.copy()

    # Amplitudes
    temp_scaling_amps = np.load(ks_dir / "amplitudes.npy").flatten().astype(np.float64)

    # PC features
    pc_feat: np.ndarray | None = None
    pc_feat_ind: np.ndarray | None = None
    if load_pcs:
        pc_feat = np.load(ks_dir / "pc_features.npy").astype(np.float64)
        pc_feat_ind = np.load(ks_dir / "pc_feature_ind.npy").astype(np.int64)

    # Cluster groups
    cgs_file = None
    for name in ("cluster_groups.csv", "cluster_group.tsv"):
        candidate = ks_dir / name
        if candidate.exists():
            cgs_file = candidate
    # Also check KS4-specific label file
    ks_label_file = ks_dir / "cluster_KSLabel.tsv"

    if cgs_file is not None:
        cids, cgs = read_cluster_groups(cgs_file)
    elif ks_label_file.exists():
        cids, cgs = read_cluster_groups(ks_label_file)
    else:
        cids = np.unique(spike_templates)
        cgs = np.full_like(cids, 3)  # unsorted

    # Exclude noise clusters
    if exclude_noise and len(cids) > 0:
        noise_clusters = cids[cgs == 0]
        if len(noise_clusters) > 0:
            keep = ~np.isin(clu, noise_clusters)
            st = st[keep]
            spike_templates = spike_templates[keep]
            temp_scaling_amps = temp_scaling_amps[keep]
            if pc_feat is not None:
                pc_feat = pc_feat[keep]
            clu = clu[keep]
            cgs = cgs[~np.isin(cids, noise_clusters)]
            cids = cids[~np.isin(cids, noise_clusters)]

    # Channel positions
    coords = np.load(ks_dir / "channel_positions.npy").astype(np.float64)
    xcoords = coords[:, 0]
    ycoords = coords[:, 1]

    # Channel map (0-indexed)
    chan_map = np.load(ks_dir / "channel_map.npy").flatten().astype(np.int64)

    # Templates
    temps = np.load(ks_dir / "templates.npy").astype(np.float64)

    # Whitening matrix inverse
    winv = np.load(ks_dir / "whitening_mat_inv.npy").astype(np.float64)

    return KilosortOutput(
        st=st,
        spike_templates=spike_templates,
        clu=clu,
        temp_scaling_amps=temp_scaling_amps,
        cgs=cgs,
        cids=cids,
        xcoords=xcoords,
        ycoords=ycoords,
        temps=temps,
        winv=winv,
        pc_feat=pc_feat,
        pc_feat_ind=pc_feat_ind,
        chan_map=chan_map,
        sample_rate=sample_rate,
    )
