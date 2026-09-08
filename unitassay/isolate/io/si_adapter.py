"""SpikeInterface I/O adapter.

Converts between SpikeInterface extractors and mind_snag data types.
Requires ``pip install mind-snag[spikeinterface]``.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from unitassay.isolate.types import KilosortOutput, NPcluData

logger = logging.getLogger(__name__)

try:
    import spikeinterface as si
    import spikeinterface.extractors as se

    HAS_SI = True
except ImportError:
    HAS_SI = False


def _check_si() -> None:
    if not HAS_SI:
        raise ImportError(
            "SpikeInterface is not installed. "
            "Install with: pip install 'mind-snag[spikeinterface]'"
        )


def recording_from_path(path: str | Path, **kwargs) -> "si.BaseRecording":
    """Load a recording from any format SpikeInterface supports.

    Parameters
    ----------
    path : path to recording (SpikeGLX, Open Ephys, NWB, etc.)
    **kwargs : passed to spikeinterface's read function

    Returns
    -------
    SpikeInterface BaseRecording
    """
    _check_si()
    path = Path(path)
    if path.suffix == ".bin" or (path / "params.py").exists():
        return se.read_spikeglx(path.parent, **kwargs)
    return si.load(path, **kwargs)


def sorting_from_si(sorting: "si.BaseSorting") -> KilosortOutput:
    """Convert a SpikeInterface sorting extractor to KilosortOutput.

    Parameters
    ----------
    sorting : SpikeInterface BaseSorting

    Returns
    -------
    KilosortOutput with spike times and cluster assignments
    """
    _check_si()
    unit_ids = sorting.get_unit_ids()
    all_st = []
    all_clu = []
    for uid in unit_ids:
        st = sorting.get_unit_spike_train(uid, return_times=True)
        all_st.append(st)
        all_clu.append(np.full(len(st), uid))

    spike_times = np.concatenate(all_st) if all_st else np.array([])
    cluster_ids = np.concatenate(all_clu) if all_clu else np.array([], dtype=np.int64)

    sort_idx = np.argsort(spike_times)
    spike_times = spike_times[sort_idx]
    cluster_ids = cluster_ids[sort_idx]

    n_units = len(unit_ids)
    return KilosortOutput(
        st=spike_times,
        spike_templates=cluster_ids,
        clu=cluster_ids,
        temp_scaling_amps=np.ones(len(spike_times)),
        cgs=np.full(n_units, 2, dtype=np.int64),
        cids=np.array(unit_ids, dtype=np.int64),
        xcoords=np.zeros(n_units),
        ycoords=np.zeros(n_units),
        temps=np.zeros((n_units, 82, 1)),
        winv=np.eye(1),
        pc_feat=None,
        pc_feat_ind=None,
        chan_map=np.arange(n_units, dtype=np.int64),
        sample_rate=sorting.get_sampling_frequency(),
    )


def to_si_sorting(
    npclu: NPcluData,
    sample_rate: float = 30000.0,
) -> "si.BaseSorting":
    """Export mind_snag NPclu data as a SpikeInterface sorting extractor.

    Parameters
    ----------
    npclu : NPcluData from unitassay.isolate
    sample_rate : sampling frequency in Hz

    Returns
    -------
    SpikeInterface NumpySorting
    """
    _check_si()
    from spikeinterface.core import NumpySorting

    spike_samples = (npclu.spike_times * sample_rate).astype(np.int64)
    labels = npclu.cluster_ids

    return NumpySorting.from_times_labels(
        times_list=spike_samples,
        labels_list=labels,
        sampling_frequency=sample_rate,
    )
