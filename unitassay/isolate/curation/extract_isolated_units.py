"""Identify isolated units from SortData and update NPclu.

Ports extract_isolated_units.m. Scans SortData files for units
marked as isolated (unit_iso == 1), extracts their spikes, and
saves IsoClu_info back into the NPclu file.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from unitassay.isolate.config import IsolateConfig
from unitassay.isolate.io.mat_reader import load_mat
from unitassay.isolate.io.hdf5_writer import write_hdf5

logger = logging.getLogger(__name__)


def extract_isolated_units(
    cfg: IsolateConfig,
    ks_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Identify isolated units and update NPclu with isolation info.

    Parameters
    ----------
    cfg : pipeline configuration
    ks_dir : path to Kilosort output directory
    output_dir : path for pipeline output (contains NPclu.h5 and SortData files)
    """
    ks_dir = Path(ks_dir)
    output_dir = Path(output_dir)

    # Load NPclu
    npclu_h5 = output_dir / "NPclu.h5"

    if npclu_h5.exists():
        npclu_data = _load_npclu_h5(npclu_h5)
    else:
        logger.warning("NPclu not found in %s", output_dir)
        return

    # NPclu stored [2 x N], view as [N x 2]: col 0 = spike times, col 1 = cluster IDs
    npclu_arr = npclu_data["NPclu"]  # already transposed in _load_npclu_h5
    spike_times = npclu_arr[:, 0]
    cluster_ids = npclu_arr[:, 1].astype(int)

    # Clu_info is [N x 2]: col 0 = cluster IDs, col 1 = channel indices
    clu_info = npclu_data["Clu_info"]

    unique_clus = np.unique(cluster_ids).astype(int)
    iso_clu_ids: list[int] = []

    for clu_id in unique_clus:
        sort_file = output_dir / f"{int(clu_id)}.SortData.h5"
        sort_file_mat = output_dir / f"{int(clu_id)}.SortData.mat"

        if sort_file.exists():
            with h5py.File(sort_file, "r") as f:
                if "frame_0000" in f:
                    unit_iso = f["frame_0000"].attrs.get("unit_iso", 0)
                else:
                    unit_iso = 0
            if unit_iso == 1:
                iso_clu_ids.append(int(clu_id))
        elif sort_file_mat.exists():
            sd = load_mat(sort_file_mat)
            sort_data = sd.get("SortData")
            if sort_data is not None:
                ui = _get_unit_iso(sort_data)
                if ui == 1:
                    iso_clu_ids.append(int(clu_id))
        else:
            logger.debug("SortData not found: %s", sort_file)

    # Build isolated spike data
    iso_mask = np.isin(cluster_ids, iso_clu_ids)
    iso_spike_times = spike_times[iso_mask]
    iso_cluster_ids = cluster_ids[iso_mask].astype(np.float64)

    # Build IsoClu_info [N x 2] — same format as Clu_info (in-memory view)
    if iso_clu_ids:
        iso_clu_list = []
        for c in iso_clu_ids:
            match = np.where(clu_info[:, 0] == c)[0]
            if len(match) > 0:
                iso_clu_list.append([c, clu_info[match[0], 1]])
        iso_clu_info = np.array(iso_clu_list, dtype=np.float64) if iso_clu_list else np.empty((0, 2), dtype=np.float64)
    else:
        iso_clu_info = np.empty((0, 2), dtype=np.float64)

    # Save updated NPclu with isolation data
    _update_npclu_with_iso(npclu_h5, npclu_data, iso_spike_times, iso_cluster_ids, iso_clu_info)

    logger.info("Saved %d isolated units to %s", len(iso_clu_ids), npclu_h5)


def _load_npclu_h5(path: Path) -> dict:
    """Load NPclu from HDF5 (MATLAB-format names).

    HDF5 file stores arrays in MATLAB column-major layout. We transpose
    2D arrays so the in-memory view matches MATLAB display:
      NPclu:      [2 x N] on disk -> [N x 2] in memory
      Clu_info:   [2 x nClu] on disk -> [nClu x 2] in memory
    """
    with h5py.File(path, "r") as f:
        data = {
            "NPclu": np.array(f["NPclu"]).T,
            "NPtemplate": np.array(f["NPtemplate"]),  # 3D, no transpose
            "Clu_info": np.array(f["Clu_info"]).T,
            "KSclu_info": np.array(f["KSclu_info"]).T,
            "tempScalingAmps": np.array(f["tempScalingAmps"]),
            "KSversion": int(f.attrs.get("ks_version", 4)),
        }
        if "pcFeat" in f:
            data["pcFeat"] = np.array(f["pcFeat"])
        return data


def _get_unit_iso(sort_data) -> int:
    """Extract UnitIso from a MATLAB SortData struct."""
    if isinstance(sort_data, dict):
        return int(sort_data.get("UnitIso", 0))
    if isinstance(sort_data, np.ndarray) and sort_data.size > 0:
        first = sort_data.flat[0]
        if hasattr(first, "UnitIso"):
            return int(first.UnitIso)
        if isinstance(first, dict):
            return int(first.get("UnitIso", 0))
    if hasattr(sort_data, "UnitIso"):
        val = sort_data.UnitIso
        return int(val.flat[0]) if isinstance(val, np.ndarray) else int(val)
    return 0


def _update_npclu_with_iso(
    path: Path, npclu_data: dict,
    iso_spike_times: np.ndarray, iso_cluster_ids: np.ndarray,
    iso_clu_info: np.ndarray,
) -> None:
    """Write updated NPclu with isolation data (MATLAB-format names).

    npclu_data has arrays in [N x 2] / [N x ...] view (transposed at load).
    Transpose back to [2 x N] for MATLAB-compatible disk storage.
    """
    # NPisoclu in [M x 2] view, transpose to [2 x M] for disk
    npisoclu_view = (
        np.column_stack([iso_spike_times, iso_cluster_ids])
        if len(iso_spike_times) > 0 else np.empty((0, 2))
    )

    data = {
        "NPclu": npclu_data["NPclu"].T,                # back to [2 x N]
        "NPtemplate": npclu_data["NPtemplate"],
        "Clu_info": npclu_data["Clu_info"].T,          # back to [2 x nClu]
        "KSclu_info": npclu_data["KSclu_info"].T,      # back to [2 x nGood]
        "tempScalingAmps": npclu_data["tempScalingAmps"],
        "NPisoclu": npisoclu_view.T,                    # back to [2 x M]
        "IsoClu_info": iso_clu_info.T,                  # back to [2 x nIso]
    }
    if "pcFeat" in npclu_data:
        data["pcFeat"] = npclu_data["pcFeat"]

    write_hdf5(
        path, data,
        attrs={"ks_version": npclu_data.get("KSversion", 4)},
    )
