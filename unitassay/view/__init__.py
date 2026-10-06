"""unitassay.view — a read-only browser viewer for unitassay's output.

The successor to the lab's MATLAB Spike_sorting_GUI (Qiao (Betty) Xu), which
showed layer 1 only. This one shows layer 1 -- per-frame PC clouds, waveforms,
L-ratio against the reference channel and each neighbour, ACG and ISI
violations -- beside layer 2's verdict for the same cluster, so a reader can
see WHY a cluster passed or failed.

    unitassay view <out_dir>

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
from .data import Run, Session        # noqa: F401
