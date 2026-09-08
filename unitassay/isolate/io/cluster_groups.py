"""Read cluster_groups.csv/tsv files from Kilosort/Phy output.

Ports readClusterGroupsCSV.m. These files map cluster IDs to quality labels:
  0 = noise, 1 = mua, 2 = good, 3 = unsorted

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# Label string -> integer mapping
_LABEL_MAP = {
    "noise": 0,
    "mua": 1,
    "good": 2,
    "unsorted": 3,
}


def read_cluster_groups(path: str | Path) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Read a cluster_groups.csv or cluster_group.tsv file.

    Parameters
    ----------
    path : path to the CSV/TSV file

    Returns
    -------
    cids : array of cluster IDs (0-indexed, as in KS output)
    cgs  : array of cluster group integers (0=noise, 1=mua, 2=good, 3=unsorted)
    """
    cids_list: list[int] = []
    cgs_list: list[int] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header lines
            parts = line.replace(",", "\t").split("\t")
            if len(parts) < 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue  # header row
            label = parts[1].strip().lower()
            cg = _LABEL_MAP.get(label, 3)
            cids_list.append(cid)
            cgs_list.append(cg)

    return np.array(cids_list, dtype=np.int64), np.array(cgs_list, dtype=np.int64)
