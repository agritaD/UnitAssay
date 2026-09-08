"""SpikeInterface quality metrics integration.

Wraps SpikeInterface's quality metric computation to provide standardized
metrics (ISI violations, SNR, presence ratio, etc.) compatible with
mind_snag's curation workflow.

Requires ``pip install mind-snag[spikeinterface]``.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import spikeinterface as si
    import spikeinterface.qualitymetrics as sqm

    HAS_SI = True
except ImportError:
    HAS_SI = False


def compute_quality_metrics(
    sorting: "si.BaseSorting",
    recording: "si.BaseRecording | None" = None,
    metric_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute quality metrics via SpikeInterface.

    Parameters
    ----------
    sorting : SpikeInterface sorting extractor
    recording : optional recording extractor (needed for waveform-based metrics)
    metric_names : list of metric names to compute. Defaults to a standard set.

    Returns
    -------
    DataFrame with one row per unit and metric columns.
    """
    if not HAS_SI:
        raise ImportError(
            "SpikeInterface is not installed. "
            "Install with: pip install 'mind-snag[spikeinterface]'"
        )

    if metric_names is None:
        metric_names = [
            "isi_violations_ratio",
            "presence_ratio",
            "firing_rate",
            "num_spikes",
        ]
        if recording is not None:
            metric_names.extend(["snr", "amplitude_median"])

    if recording is not None:
        analyzer = si.create_sorting_analyzer(sorting, recording)
        analyzer.compute("random_spikes")
        analyzer.compute("waveforms")
        analyzer.compute("templates")
        analyzer.compute("noise_levels")
        analyzer.compute("quality_metrics", metric_names=metric_names)
        return analyzer.get_extension("quality_metrics").get_data()

    # Without recording, only spike-train metrics are available
    analyzer = si.create_sorting_analyzer(sorting, recording=None)
    spike_only = [m for m in metric_names
                  if m in ("isi_violations_ratio", "presence_ratio",
                           "firing_rate", "num_spikes")]
    if spike_only:
        analyzer.compute("quality_metrics", metric_names=spike_only)
        return analyzer.get_extension("quality_metrics").get_data()

    return pd.DataFrame()
