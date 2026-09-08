"""Stitching backend abstraction.

Provides the native dual-evidence (FR + WF correlation) stitching backend.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from unitassay.isolate.config import MindSnagConfig
from unitassay.isolate.types import StitchResult

logger = logging.getLogger(__name__)


class StitchBackend(Protocol):
    """Protocol for stitching backends."""

    def run(
        self,
        cfg: MindSnagConfig,
        output_dirs: list[str | Path],
        ks_dirs: list[str | Path],
        stitch_output_dir: str | Path,
        NPelec: dict | None = None,
    ) -> StitchResult: ...


class NativeBackend:
    """mind_snag's dual-evidence (FR + WF correlation) stitching."""

    def run(
        self,
        cfg: MindSnagConfig,
        output_dirs: list[str | Path],
        ks_dirs: list[str | Path],
        stitch_output_dir: str | Path,
        NPelec: dict | None = None,
    ) -> StitchResult:
        from unitassay.isolate.stitching.stitch_neurons import NeuronStitcher

        stitcher = NeuronStitcher(
            cfg, output_dirs, ks_dirs, stitch_output_dir, NPelec=NPelec,
        )
        return stitcher.run()


_BACKENDS: dict[str, type[StitchBackend]] = {
    "native": NativeBackend,
}


def get_backend(name: str) -> StitchBackend:
    """Get a stitching backend by name.

    Parameters
    ----------
    name : 'native'

    Returns
    -------
    StitchBackend instance
    """
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown stitching backend: {name!r}. "
            f"Available: {list(_BACKENDS.keys())}"
        )
    return cls()
