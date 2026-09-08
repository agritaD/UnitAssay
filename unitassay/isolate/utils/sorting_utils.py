"""Spike sorting utilities.

Ports sortKSSpX.m and getRasters.m.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sort_spikes_by_rt(
    rt: NDArray[np.float64],
    spike_cell: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Sort spike rasters by reaction time (ascending).

    Ports sortKSSpX.m.

    Parameters
    ----------
    rt : reaction times array
    spike_cell : list of spike time arrays (one per trial)

    Returns
    -------
    sorted_rt : sorted reaction times
    sorted_spx : spike cell array sorted by RT
    """
    if rt is None or len(rt) == 0:
        return np.array([]), spike_cell

    sort_idx = np.argsort(rt)
    sorted_rt = rt[sort_idx]
    sorted_spx = [spike_cell[i] for i in sort_idx]
    return sorted_rt, sorted_spx


def get_rasters(
    spike_cell: list[NDArray[np.float64]],
    bn: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract raster plot coordinates from spike cell array.

    Ports getRasters.m.

    Parameters
    ----------
    spike_cell : list of spike time arrays (one per trial)
    bn : (start, stop) time window in ms

    Returns
    -------
    x : all spike times (scatter x-coordinates)
    y : trial indices (scatter y-coordinates)
    """
    dt = 0.08
    start = bn[0]

    x_list = []
    y_list = []
    for i_tr, spikes in enumerate(spike_cell):
        if spikes is None or len(spikes) == 0:
            continue
        x = np.asarray(spikes).flatten() + start
        y = np.full_like(x, (i_tr + 1) * dt)
        x_list.append(x)
        y_list.append(y)

    if x_list:
        return np.concatenate(x_list), np.concatenate(y_list)
    return np.array([]), np.array([])
