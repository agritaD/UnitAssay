"""unitassay — measure, curate and link spike-sorted units.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Licence: MIT

Three layers, applied after a spike sorter (Kilosort, or any sorter that
writes the standard Phy-style arrays):

    isolate   per-frame isolation quality — the L-ratio of each unit against a
              shape-dissimilar reference channel and against each neighbouring
              cluster, written as one SortData.h5 per cluster
    curate    eight rules that decide keep/reject, a taxonomy that names the
              mechanism, and a findings layer that names the ERROR CLASS
    link      matching the same unit across recordings              (in progress)

    >>> from unitassay import curate
    >>> rows = curate.curate_session(ks4_dir, isolation_dir='isolation')

WHAT THIS TOOL CLAIMS, AND WHAT IT DOES NOT. On a synthetic corpus with ground
truth it reaches 83.7% balanced accuracy (sensitivity 81.3, specificity 86.1)
over 3,696 clusters. It is not an oracle, and its more useful property is that
it says WHICH clusters it could not decide rather than guessing: `mixed_cluster`
makes no claim about the error class, and a rule that ran without reaching a
verdict is recorded as declined rather than as clean.
"""
__version__ = '0.1.0'
__author__ = 'Agrita Dubey, Qiao (Betty) Xu'
__license__ = 'MIT'

from . import curate, isolate            # noqa: F401
