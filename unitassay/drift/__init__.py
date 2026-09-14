"""Layer 3 — drift: quantify and calibrate the motion KS4 reports.

Shipped so far: `gap_blocks`, which detects KS4's gap blocks on multi-bank
probes and EXCLUDES them from the session drift. Exclusion, not repair — see
that module's docstring. The rest of the layer (session metrics, the
calibrated floors, the artifact criteria) is specified but not yet built.
"""
from unitassay.drift.gap_blocks import (detect, exclude_gap_blocks,
                                        range_90_per_block)

__all__ = ['detect', 'exclude_gap_blocks', 'range_90_per_block']
