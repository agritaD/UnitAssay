"""Layer 3 — drift: quantify and calibrate the motion Kilosort reports.

    metrics      range_90, slow/fast, max_step, the artifact criteria
    gap_blocks   detect the multi-bank gap blocks, and EXCLUDE them

`gap_blocks` EXCLUDES; it does not repair. See its docstring.
"""
from unitassay.drift.gap_blocks import detect, exclude_gap_blocks
from unitassay.drift.metrics import (artifact_flags, load_dshift, max_step,
                                     range_90, range_90_per_block,
                                     session_metrics, slow_fast)

__all__ = ['detect', 'exclude_gap_blocks', 'load_dshift', 'range_90',
           'range_90_per_block', 'slow_fast', 'max_step', 'session_metrics',
           'artifact_flags']
