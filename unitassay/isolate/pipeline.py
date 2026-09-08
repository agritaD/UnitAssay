"""Pipeline orchestrator for spike sorting stages.

Ports pipeline_KS4.m. Uses a Stage enum set for selective stage execution.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from unitassay.isolate.config import MindSnagConfig

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Pipeline stages (matches MATLAB pipeline_KS4)."""

    KILOSORT = "kilosort"
    EXTRACT = "extract"
    ISOLATION = "isolation"
    CURATION = "curation"
    ISO_UNITS = "iso_units"


ALL_STAGES = list(Stage)

# Default stages: EXTRACT through ISO_UNITS (skip KILOSORT)
DEFAULT_STAGES = [Stage.EXTRACT, Stage.ISOLATION, Stage.CURATION, Stage.ISO_UNITS]


class Pipeline:
    """Orchestrates the full spike sorting pipeline.

    Parameters
    ----------
    cfg : MindSnagConfig instance
    """

    def __init__(self, cfg: MindSnagConfig):
        self.cfg = cfg

    def run(
        self,
        ks_dir: str | Path,
        output_dir: str | Path,
        stages: list[str | Stage] | None = None,
        bin_file: str | Path | None = None,
    ) -> None:
        """Run the pipeline.

        Parameters
        ----------
        ks_dir : path to Kilosort output directory (input for stages 1-4,
                 output destination for stage 0)
        output_dir : path for pipeline output (NPclu.mat, SortData, etc.)
        stages : list of stage names to run (default: EXTRACT through ISO_UNITS)
        bin_file : path to .ap.bin file; required for KILOSORT stage
        """
        cfg = self.cfg
        ks_dir = Path(ks_dir)
        output_dir = Path(output_dir)

        if stages is None:
            active_stages = set(DEFAULT_STAGES)
        else:
            active_stages = {Stage(s) if isinstance(s, str) else s for s in stages}

        logger.info("")
        logger.info("=== mind_snag Pipeline KS4 ===")
        logger.info("KS dir: %s", ks_dir)
        logger.info("Output dir: %s", output_dir)
        logger.info("Stages: %s", " -> ".join(
            s.value for s in ALL_STAGES if s in active_stages
        ))

        # Stage 0: Kilosort4
        if Stage.KILOSORT in active_stages:
            logger.info("--- Stage 0: Running Kilosort4 ---")
            if bin_file is None:
                raise ValueError(
                    "bin_file must be provided when running KILOSORT stage"
                )
            from unitassay.isolate.sorting.run_kilosort4 import run_kilosort4
            run_kilosort4(cfg, str(bin_file), ks_dir)
            logger.info("--- Kilosort4 complete ---")

        # Stage 1: Spike extraction
        if Stage.EXTRACT in active_stages:
            logger.info("--- Stage 1: Extracting spikes ---")
            from unitassay.isolate.sorting.extract_spikes import extract_spikes
            extract_spikes(ks_dir, output_dir)
            logger.info("--- Spike extraction complete ---")

        # Stage 2: Isolation analysis
        if Stage.ISOLATION in active_stages:
            logger.info("--- Stage 2: Computing isolation scores ---")
            from unitassay.isolate.curation.compute_isolation import compute_isolation
            compute_isolation(cfg, ks_dir, output_dir)
            logger.info("--- Isolation analysis complete ---")

        # Stage 3: Auto-curation
        if Stage.CURATION in active_stages:
            logger.info("--- Stage 3: Auto-curation ---")
            logger.info(
                "Thresholds: L-Ratio=%.2f, ISI=%.2f, t-Ratio=%.2f",
                cfg.curation.l_ratio_threshold,
                cfg.curation.isi_violation_rate,
                cfg.curation.isolated_t_ratio,
            )
            from unitassay.isolate.curation.auto_curate import auto_curate
            auto_curate(output_dir, cfg)
            logger.info("--- Auto-curation complete ---")

        # Stage 4: Isolated unit extraction
        if Stage.ISO_UNITS in active_stages:
            logger.info("--- Stage 4: Extracting isolated units ---")
            from unitassay.isolate.curation.extract_isolated_units import extract_isolated_units
            extract_isolated_units(cfg, ks_dir, output_dir)
            logger.info("--- Isolated unit extraction complete ---")

        logger.info("=== Pipeline complete ===")
