"""Convert between .mat and HDF5 formats.

Provides bidirectional conversion for NPclu, SortData, and RasterData files.

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from unitassay.screen.io.mat_reader import load_mat
from unitassay.screen.io.hdf5_writer import write_hdf5

logger = logging.getLogger(__name__)


def convert_file(
    input_path: Path,
    output_path: Path,
    direction: str = "mat2hdf5",
) -> None:
    """Convert a file between .mat and HDF5 formats.

    Parameters
    ----------
    input_path : source file
    output_path : destination file
    direction : 'mat2hdf5' or 'hdf52mat'
    """
    if direction == "mat2hdf5":
        _mat_to_hdf5(input_path, output_path)
    elif direction == "hdf52mat":
        _hdf5_to_mat(input_path, output_path)
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'mat2hdf5' or 'hdf52mat'.")


def _mat_to_hdf5(mat_path: Path, h5_path: Path) -> None:
    """Convert a .mat file to HDF5."""
    data = load_mat(mat_path)

    # Convert all numpy arrays and scalars
    h5_data = {}
    attrs = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            h5_data[key] = val
        elif isinstance(val, (int, float, str)):
            attrs[key] = val
        elif isinstance(val, np.generic):
            attrs[key] = val.item()

    attrs["source_format"] = "matlab"
    attrs["source_file"] = str(mat_path.name)

    write_hdf5(h5_path, h5_data, attrs=attrs)
    logger.info("Converted %s -> %s", mat_path, h5_path)


def _hdf5_to_mat(h5_path: Path, mat_path: Path) -> None:
    """Convert an HDF5 file to .mat (v7.3)."""
    import scipy.io

    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            item = f[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                # Flatten group to arrays
                for sub_key in item.keys():
                    if isinstance(item[sub_key], h5py.Dataset):
                        data[f"{key}_{sub_key}"] = item[sub_key][()]

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(mat_path), data, do_compression=True)
    logger.info("Converted %s -> %s", h5_path, mat_path)


def convert_npclu_mat_to_h5(mat_path: Path, h5_path: Path | None = None) -> Path:
    """Convert an NPclu.mat file specifically to the NPclu HDF5 schema.

    Parameters
    ----------
    mat_path : path to NPclu.mat
    h5_path : output path (default: same name with .h5 extension)

    Returns
    -------
    Path to output HDF5 file
    """
    if h5_path is None:
        h5_path = mat_path.with_suffix(".h5")

    data = load_mat(mat_path)
    npclu = data.get("NPclu", np.array([]))

    h5_data = {}
    if len(npclu) > 0:
        h5_data["spike_times"] = npclu[:, 0]
        h5_data["cluster_ids"] = npclu[:, 1].astype(np.int64)

    for key_map in [
        ("NPtemplate", "templates"),
        ("Clu_info", "clu_info"),
        ("KSclu_info", "ks_clu_info"),
        ("pcFeat", "pc_feat"),
        ("tempScalingAmps", "temp_scaling_amps"),
        ("NPisoclu", "iso_spike_data"),
        ("IsoClu_info", "iso_clu_info"),
    ]:
        mat_key, h5_key = key_map
        if mat_key in data:
            val = data[mat_key]
            if isinstance(val, np.ndarray):
                h5_data[h5_key] = val

    attrs = {"ks_version": int(data.get("KSversion", 4))}
    write_hdf5(h5_path, h5_data, attrs=attrs)
    logger.info("Converted NPclu: %s -> %s", mat_path, h5_path)

    return h5_path
