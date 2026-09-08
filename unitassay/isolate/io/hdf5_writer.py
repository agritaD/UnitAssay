"""Write pipeline outputs in HDF5 format with gzip compression.

Schema for NPclu.h5 (matches MATLAB NPclu.mat format):
  /NPclu             float64 [2 x N]  — row 0: spike times, row 1: cluster IDs
  /NPtemplate        float64 [nChannels x nTimePoints x nTemplates]
  /Clu_info          float64 [2 x nClusters]  — row 0: cluster IDs, row 1: channel indices
  /KSclu_info        float64 [2 x nGood]
  /pcFeat            float64 [nLocalChannels x 3 x N]
  /tempScalingAmps   float64 [1 x N]
  attrs: ks_version, creation_date

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


def write_hdf5(
    path: str | Path,
    data: dict[str, NDArray | Any],
    attrs: dict[str, Any] | None = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Write data dictionary to an HDF5 file.

    Parameters
    ----------
    path : output file path
    data : dict mapping dataset names to numpy arrays
    attrs : optional dict of file-level attributes
    compression : compression algorithm (default 'gzip')
    compression_opts : compression level (default 4)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for key, val in data.items():
            if val is None:
                continue
            arr = np.asarray(val)
            f.create_dataset(
                key, data=arr,
                compression=compression,
                compression_opts=compression_opts,
            )

        # Default attributes
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["format"] = "mind_snag_v2"

        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v


def write_npclu_h5(
    path: str | Path,
    spike_times: NDArray,
    cluster_ids: NDArray,
    templates: NDArray,
    clu_info: NDArray,
    ks_clu_info: NDArray,
    pc_feat: NDArray | None,
    temp_scaling_amps: NDArray,
    ks_version: int = 4,
) -> None:
    """Write NPclu data in HDF5 format matching MATLAB NPclu.mat layout.

    Data is stored in MATLAB-compatible format:
    - NPclu: [2 x N] (spike_times; cluster_ids)
    - Clu_info: [2 x nClusters] (cluster_ids; channel_indices)
    - NPtemplate: [nChannels x nTimePoints x nTemplates]
    - pcFeat: [nLocalChannels x 3 x N]
    - tempScalingAmps: [1 x N]

    Parameters
    ----------
    path : output file path
    spike_times : spike times [N]
    cluster_ids : cluster ID per spike [N]
    templates : template waveforms [nTemplates x nTimePoints x nChannels]
    clu_info : cluster-to-channel mapping [nClusters x 2]
    ks_clu_info : good units mapping [nGood x 2]
    pc_feat : PC features [N x 3 x nLocalChannels] (or None)
    temp_scaling_amps : template scaling amplitudes [N]
    ks_version : Kilosort version
    """
    # Convert to MATLAB layout (transposed)
    npclu = np.vstack([spike_times.astype(np.float64),
                       cluster_ids.astype(np.float64)])  # [2 x N]

    data = {
        "NPclu": npclu,
        "NPtemplate": templates.transpose(2, 1, 0).astype(np.float32),  # [nChan x nTime x nTempl]
        "Clu_info": clu_info.T.astype(np.float64),       # [2 x nClusters]
        "KSclu_info": ks_clu_info.T.astype(np.float64),  # [2 x nGood]
        "tempScalingAmps": temp_scaling_amps.reshape(1, -1).astype(np.float32),  # [1 x N]
    }
    if pc_feat is not None:
        data["pcFeat"] = pc_feat.transpose(2, 1, 0).astype(np.float32)  # [nLocal x 3 x N]

    write_hdf5(
        path, data,
        attrs={"ks_version": ks_version},
    )


def write_sort_data_h5(
    path: str | Path,
    frames: list[dict[str, Any]],
) -> None:
    """Write SortData (isolation analysis) in HDF5 format.

    Parameters
    ----------
    path : output file path
    frames : list of dicts, one per time window, with keys matching SortDataFrame fields
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["format"] = "mind_snag_v2"
        f.attrs["n_frames"] = len(frames)

        for i, frame in enumerate(frames):
            grp = f.create_group(f"frame_{i:04d}")
            for key, val in frame.items():
                if val is None:
                    continue
                if isinstance(val, (int, float)):
                    grp.attrs[key] = val
                elif isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val, compression="gzip")
                elif isinstance(val, list):
                    # List of arrays (e.g., Other)
                    sub = grp.create_group(key)
                    for j, arr in enumerate(val):
                        if arr is not None and isinstance(arr, np.ndarray):
                            sub.create_dataset(f"{j:04d}", data=arr, compression="gzip")


def write_raster_data_h5(
    path: str | Path,
    clu: int,
    spike_clu: list,
    rt: NDArray | None,
    other_clu: NDArray | None = None,
    other_spike_clu: list | None = None,
    other_rt: list | None = None,
) -> None:
    """Write RasterData in HDF5 format.

    Parameters
    ----------
    path : output file path
    clu : cluster ID
    spike_clu : list of spike time arrays (one per trial)
    rt : reaction times array
    other_clu : neighboring cluster IDs
    other_spike_clu : neighboring cluster spikes
    other_rt : neighboring cluster reaction times
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["creation_date"] = datetime.now().isoformat()
        f.attrs["format"] = "mind_snag_v2"
        f.attrs["clu"] = clu

        # Write spike times per trial as variable-length datasets
        trials_grp = f.create_group("spike_clu")
        for i, spikes in enumerate(spike_clu):
            arr = np.asarray(spikes if spikes is not None else [])
            trials_grp.create_dataset(f"{i:06d}", data=arr.flatten())

        if rt is not None:
            f.create_dataset("rt", data=np.asarray(rt))

        if other_clu is not None:
            f.create_dataset("other_clu", data=np.asarray(other_clu))

        if other_spike_clu is not None:
            other_grp = f.create_group("other_spike_clu")
            for i, clu_data in enumerate(other_spike_clu):
                clu_sub = other_grp.create_group(f"{i:04d}")
                if clu_data is not None:
                    for j, spikes in enumerate(clu_data if isinstance(clu_data, list) else [clu_data]):
                        arr = np.asarray(spikes if spikes is not None else [])
                        clu_sub.create_dataset(f"{j:06d}", data=arr.flatten())
