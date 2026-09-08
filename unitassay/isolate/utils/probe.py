"""Neuropixel probe detection from experiment metadata.

Ports get_neuropixel_microdrives.m.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import re
from typing import Any


# Known Neuropixel probe types
_NP_TYPES = {"np4l", "neuropixel", "np1", "np6", "modularv1"}
_NP_NAME_PATTERN = re.compile(r"NP|neuropix|modular", re.IGNORECASE)


def get_neuropixel_microdrives(
    experiment: dict[str, Any],
) -> tuple[list[dict], list[int]]:
    """Filter experiment hardware to find Neuropixel probes.

    Parameters
    ----------
    experiment : experiment dict from load_experiment()

    Returns
    -------
    np_drives : list of microdrive dicts that are Neuropixel probes
    np_indices : their indices in the original microdrive list
    """
    np_drives: list[dict] = []
    np_indices: list[int] = []

    if not experiment or "hardware" not in experiment:
        return np_drives, np_indices

    hardware = experiment["hardware"]
    if not isinstance(hardware, dict) or "microdrive" not in hardware:
        return np_drives, np_indices

    drives = hardware["microdrive"]
    if not isinstance(drives, list):
        drives = [drives]

    for i, drive in enumerate(drives):
        if not isinstance(drive, dict):
            continue
        is_np = False
        drive_type = drive.get("type", "")
        if isinstance(drive_type, str) and drive_type.lower() in _NP_TYPES:
            is_np = True
        if not is_np:
            drive_name = drive.get("name", "")
            if isinstance(drive_name, str) and _NP_NAME_PATTERN.search(drive_name):
                is_np = True
        if is_np:
            np_drives.append(drive)
            np_indices.append(i)

    return np_drives, np_indices
