"""Extract spike times from Kilosort output.

Ports extract_spikes.m. Loads KS results and saves structured cluster data.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from unitassay.screen.io.ks_loader import load_ks_dir
from unitassay.screen.io.cluster_groups import read_cluster_groups
from unitassay.screen.io.hdf5_writer import write_npclu_h5
from unitassay.screen.utils.channel_info import clus_channel_info

logger = logging.getLogger(__name__)


def extract_spikes(
    ks_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Extract spike times from Kilosort output.

    Parameters
    ----------
    ks_dir : path to Kilosort output directory
    output_dir : path for pipeline output
    """
    ks_dir = Path(ks_dir)
    output_dir = Path(output_dir)

    if not ks_dir.exists():
        raise FileNotFoundError(
            f"Kilosort output directory not found: {ks_dir}\nRun run_kilosort4 first."
        )

    # Load KS output
    sp = load_ks_dir(ks_dir, exclude_noise=False)
    max_site, _, _ = clus_channel_info(ks_dir)

    # Build NPclu — use spike times directly from KS output
    npclu = np.column_stack([sp.st, sp.spike_templates + 1])  # 0->1 indexing
    templates = sp.temps
    pc_feat = sp.pc_feat[:, :3, :] if sp.pc_feat is not None else None
    temp_scaling = sp.temp_scaling_amps

    # Cluster info
    clu_id = np.unique(sp.clu) + 1
    clu_info = np.column_stack([clu_id, max_site[: len(clu_id)]])

    # KS good units
    ks_file = ks_dir / "cluster_KSLabel.tsv"
    cids, cgs = read_cluster_groups(ks_file)
    ks_clu_id = cids[cgs == 2] + 1
    ks_mask = np.isin(clu_info[:, 0], ks_clu_id)
    ks_clu_info = clu_info[ks_mask]

    # Save
    out_file = output_dir / "NPclu.h5"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving: %s", out_file)
    write_npclu_h5(
        out_file,
        spike_times=npclu[:, 0],
        cluster_ids=npclu[:, 1].astype(np.int64),
        templates=templates,
        clu_info=clu_info,
        ks_clu_info=ks_clu_info,
        pc_feat=pc_feat,
        temp_scaling_amps=temp_scaling,
        ks_version=4,
    )
