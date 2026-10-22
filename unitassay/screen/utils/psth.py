"""Peri-stimulus time histogram with Gaussian smoothing.

Ports psth.m. Computes firing rate from trial-aligned spike times.

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def psth(
    spike_cell: list[NDArray[np.float64]],
    bn: tuple[int, int],
    smoothing: float = 50.0,
    max_rate: float = 50.0,
    spike_time_offset: float = 0.0,
) -> tuple[NDArray[np.float64], int]:
    """Compute PSTH from a list of per-trial spike times.

    Parameters
    ----------
    spike_cell : list of arrays, each containing per-trial spike times (ms)
    bn : (start, stop) time window in ms (relative to event)
    smoothing : Gaussian smoothing std in ms
    max_rate : not used in computation, kept for API compat
    spike_time_offset : offset (ms) of the event in raw trial spike times.
        - Set to 0 if spike times are already event-aligned (e.g., negative
          values for pre-event spikes). This is the standard convention.
        - Set to e.g. 300 if spike times are raw trial times with the event
          at +300 ms within each trial. Spike times will be aligned by
          subtracting this offset.

    Returns
    -------
    rate : smoothed firing rate vector (spikes/s), length = stop - start + 1
    n_tr : number of trials
    """
    n_tr = len(spike_cell)
    if n_tr == 0:
        return np.zeros(bn[1] - bn[0] + 1), 0

    start, stop = bn

    # Collect all spikes, subtracting offset to make them event-aligned
    all_spikes = []
    for spikes in spike_cell:
        if spikes is not None and len(spikes) > 0:
            arr = np.asarray(spikes).flatten() - spike_time_offset
            all_spikes.append(arr)
    if all_spikes:
        xx = np.concatenate(all_spikes)
    else:
        xx = np.array([])

    # Histogram with bin centers at start:stop (matches MATLAB hist semantics)
    centers = np.arange(start, stop + 1, dtype=np.float64)
    edges = np.concatenate([
        [-np.inf],
        (centers[:-1] + centers[1:]) / 2.0,
        [np.inf],
    ])
    z, _ = np.histogram(xx, bins=edges)
    z = z.astype(np.float64)

    # Gaussian smoothing kernel
    half_width = int(3 * smoothing)
    kernel_x = np.arange(-half_width, half_width + 1, dtype=np.float64)
    window = norm.pdf(kernel_x, 0, smoothing)

    # Convolve and trim
    rate = (1000.0 / n_tr) * np.convolve(z, window)
    rate = rate[half_width: half_width + len(centers)]

    return rate, n_tr
