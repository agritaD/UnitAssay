"""Read .mat files (v7 and v7.3/HDF5 format).

Auto-detects format and uses scipy.io.loadmat for v7,
h5py for v7.3 (HDF5-based).

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_mat(path: str | Path, variable: str | None = None) -> dict[str, Any]:
    """Load a .mat file, auto-detecting v7 vs v7.3.

    Parameters
    ----------
    path : path to .mat file
    variable : if specified, load only this variable

    Returns
    -------
    dict mapping variable names to values (numpy arrays or nested dicts)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    if _is_hdf5(path):
        return _load_mat_v73(path, variable)
    else:
        return _load_mat_v7(path, variable)


def _is_hdf5(path: Path) -> bool:
    """Check if file is HDF5 (raw HDF5 or MATLAB v7.3).

    MATLAB v7.3 files start with a 128-byte text header
    ("MATLAB 7.3 MAT-file, ..."), so we check for either:
    - Raw HDF5 magic bytes at the start (\\x89HDF)
    - "MATLAB 7.3" in the first 128 bytes
    """
    with open(path, "rb") as f:
        header = f.read(128)
    if header[:4] == b"\x89HDF":
        return True
    if b"MATLAB 7.3" in header:
        return True
    return False


def _load_mat_v7(path: Path, variable: str | None) -> dict[str, Any]:
    """Load MAT v5/v7 file via scipy."""
    import scipy.io

    kwargs: dict[str, Any] = {"squeeze_me": True, "struct_as_record": False}
    if variable:
        kwargs["variable_names"] = [variable]
    data = scipy.io.loadmat(str(path), **kwargs)
    # Filter out metadata keys
    return {k: v for k, v in data.items() if not k.startswith("__")}


def _load_mat_v73(path: Path, variable: str | None) -> dict[str, Any]:
    """Load MAT v7.3 (HDF5) file via h5py."""
    import h5py

    result: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        keys = [variable] if variable and variable in f else list(f.keys())
        # Skip '#refs#' and other HDF5 metadata groups
        keys = [k for k in keys if not k.startswith("#")]
        for key in keys:
            result[key] = _h5_to_numpy(f[key], f)
    return result


def _h5_to_numpy(item: Any, root_file: Any) -> Any:
    """Recursively convert HDF5 items to numpy arrays or dicts.

    Handles MATLAB v7.3 quirks:
    - Cell arrays stored as arrays of HDF5 object references
    - Empty arrays stored with MATLAB_empty=1 attribute (data is just the shape)
    - Strings stored as uint16 arrays
    - Column-major (Fortran) order for 2D arrays
    """
    import h5py

    if isinstance(item, h5py.Dataset):
        # MATLAB empty arrays have MATLAB_empty=1 attribute
        if item.attrs.get("MATLAB_empty", 0) == 1:
            return np.array([])

        data = item[()]

        # MATLAB cell arrays are stored as arrays of object references.
        # Dereference each one to get the actual data.
        if data.dtype == h5py.special_dtype(ref=h5py.Reference) or \
           (hasattr(data, 'dtype') and data.dtype.kind == 'O' and
            data.size > 0 and isinstance(data.flat[0], h5py.h5r.Reference)):
            result = []
            for ref in data.flat:
                if ref:  # skip null references
                    deref = root_file[ref]
                    val = _h5_to_numpy(deref, root_file)
                    result.append(val)
                else:
                    result.append(np.array([]))
            return np.array(result, dtype=object)

        # MATLAB stores strings as uint16 arrays in v7.3
        if data.dtype == np.uint16:
            try:
                return "".join(chr(c) for c in data.flatten())
            except (UnicodeDecodeError, TypeError):
                pass

        # MATLAB uses column-major (Fortran) order; transpose 2D arrays
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data.T
        return data
    elif isinstance(item, h5py.Group):
        # Check if it's a MATLAB struct
        result = {}
        for key in item.keys():
            result[key] = _h5_to_numpy(item[key], root_file)
        return result
    else:
        return item
