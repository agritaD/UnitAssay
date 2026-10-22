"""Configuration dataclasses and YAML serialization.

Ports mind_snag_config.m to Python nested dataclasses with YAML support.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CurationConfig:
    """Thresholds for auto-curation."""

    l_ratio_threshold: float = 0.2
    isi_violation_rate: float = 0.2
    isolated_t_ratio: float = 0.6


@dataclass
class RasterConfig:
    """Parameters for raster/PSTH extraction."""

    time_window: tuple[int, int] = (-300, 500)
    smoothing: int = 10
    # Offset (ms) of the event in raw trial spike times. Set to 0 if your
    # RasterData spike times are already event-aligned (negative for pre-event).
    # Set to e.g. 300 if your spike times are raw trial times with the event
    # at +300 ms.
    spike_time_offset: float = 0.0


@dataclass
class IsolationConfig:
    """Parameters for isolation scoring."""

    window_sec: int = 100
    # EXPERIMENTAL (dev copy only): max cosine similarity, in [0, 1], allowed
    # between a candidate noise channel's waveform and the unit's own peak-
    # channel waveform before it's excluded as decay bleed-through rather
    # than independent noise. See clus_channel_info() in utils/channel_info.py.
    noise_shape_sim_threshold: float = 0.5


@dataclass
class IsolateConfig:
    """Central configuration for the mind_snag pipeline.

    All pipeline functions take this config as their first argument,
    replacing the old MATLAB global variables.
    """

    gpu: int = 0  # 0-indexed (MATLAB was 1-indexed)
    n_threads: int = 64
    ks_version: int = 4
    probe_file: str = ""          # path to probe file (.mat, .prb, or .json)
    ks_params_file: str = ""      # path to KS4 params JSON (empty = use KS4 defaults)
    n_chan_bin: int = 385

    curation: CurationConfig = field(default_factory=CurationConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    isolation: IsolationConfig = field(default_factory=IsolationConfig)

    def __post_init__(self) -> None:
        # Convert nested dicts from YAML loading to dataclass instances
        if isinstance(self.curation, dict):
            self.curation = CurationConfig(**self.curation)
        if isinstance(self.raster, dict):
            raster_dict = dict(self.raster)
            if "time_window" in raster_dict:
                raster_dict["time_window"] = tuple(raster_dict["time_window"])
            self.raster = RasterConfig(**raster_dict)
        if isinstance(self.isolation, dict):
            self.isolation = IsolationConfig(**self.isolation)

    @classmethod
    def from_yaml(cls, path: str | Path) -> IsolateConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Older configs carry a `stitching:` section for a feature that is no
        # longer part of the package; ignore it rather than fail.
        data.pop("stitching", None)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        d = _config_to_dict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(obj: Any) -> Any:
    """Recursively convert config dataclasses to dicts for YAML."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = _config_to_dict(val)
        return result
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj
