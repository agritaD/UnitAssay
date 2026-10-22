"""Shared dataclasses for pipeline data structures.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class KilosortOutput:
    """Data loaded from a Kilosort output directory.

    All indices are 0-indexed (converted from MATLAB 1-indexing on load).
    """

    st: NDArray[np.float64]             # spike times in seconds
    spike_templates: NDArray[np.int64]  # template assignment per spike (0-indexed)
    clu: NDArray[np.int64]              # cluster assignment per spike
    temp_scaling_amps: NDArray[np.float64]
    cgs: NDArray[np.int64]              # cluster groups (0=noise,1=mua,2=good,3=unsorted)
    cids: NDArray[np.int64]             # cluster IDs
    xcoords: NDArray[np.float64]
    ycoords: NDArray[np.float64]
    temps: NDArray[np.float64]          # template waveforms [nTemplates x nTimePoints x nChannels]
    winv: NDArray[np.float64]           # whitening matrix inverse
    pc_feat: NDArray[np.float64] | None  # [nSpikes x nFeatures x nLocalChannels]
    pc_feat_ind: NDArray[np.int64] | None  # [nTemplates x nLocalChannels]
    chan_map: NDArray[np.int64]          # channel map (0-indexed)
    sample_rate: float = 30000.0


@dataclass
class NPcluData:
    """Drift-corrected spike data (from NPclu.mat or NPclu.h5).

    NPclu matrix columns: [spike_time, cluster_id].
    Cluster IDs are 1-indexed in MATLAB output, 0-indexed in Python.
    """

    spike_times: NDArray[np.float64]    # drift-corrected spike times
    cluster_ids: NDArray[np.int64]      # cluster ID per spike
    templates: NDArray[np.float64]      # template waveforms
    clu_info: NDArray[np.int64]         # [cluster_id, channel_index] mapping
    ks_clu_info: NDArray[np.int64]      # good units only
    pc_feat: NDArray[np.float64] | None
    temp_scaling_amps: NDArray[np.float64]
    ks_version: int = 4
    # Optional fields set by extract_isolated_units
    iso_spike_times: NDArray[np.float64] | None = None
    iso_cluster_ids: NDArray[np.int64] | None = None
    iso_clu_info: NDArray[np.int64] | None = None


@dataclass
class SortDataFrame:
    """Isolation data for one time window of one cluster."""

    clu_wf: NDArray[np.float64]              # cluster waveform
    noise_wf: NDArray[np.float64]            # noise channel waveform
    unit: NDArray[np.float64] | None = None  # unit PC features
    noise: NDArray[np.float64] | None = None # noise PC features
    mean_spike_amp: NDArray[np.float64] | None = None
    mean_noise_amp: NDArray[np.float64] | None = None
    sd_noise_amp: NDArray[np.float64] | None = None
    score: float | None = None
    unit_iso: int = 0
    clu: int = 0
    other: list[NDArray[np.float64]] | None = None
    other_clu: NDArray[np.int64] | None = None
    other_good_bad: NDArray[np.int64] | None = None
    other_clu_wf: NDArray[np.float64] | None = None


@dataclass
class RasterDataEntry:
    """Trial-aligned raster data for one cluster."""

    clu: int
    spike_clu: list[NDArray[np.float64]]      # spike times per trial
    other_clu: NDArray[np.int64] | None       # neighboring cluster IDs
    other_spike_clu: list[Any] | None         # neighboring cluster spikes
    rt: NDArray[np.float64] | None            # reaction times
    other_rt: list[NDArray[np.float64]] | None = None
