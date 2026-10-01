"""unitassay — measure and curate spike-sorted units.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Licence: MIT

Three layers, applied after a spike sorter (Kilosort, or any sorter that
writes the standard Phy-style arrays):

    screen    per-frame isolation quality — the L-ratio of each unit against a
              shape-dissimilar reference channel and against each neighbouring
              cluster, written as one SortData.h5 per cluster
    curate    five tests and three evidence guards that decide each cluster's
              OUTCOME -- PASS, FAIL or LOW_SPIKE_COUNT, nothing else -- and a
              findings layer that names the ERROR CLASS of every FAIL
    drift     quantify and calibrate the drift KS4 reports, and FLAG A KS4 BUG:
              KS4 splits its drift blocks by depth BIN INDEX with no reference
              to the channel map, so on a multi-bank probe blocks land in the
              dead gap between banks and report interpolation, not measurement
              — 5.0x inflation with a true drift of exactly ZERO, affecting 72
              of our 147 recordings.  gap_blocks.detect and
              gap_blocks.exclude_gap_blocks are BUILT and reproduce the
              147-session analysis 147/147. They EXCLUDE the affected
              blocks; they do not repair the bug, which needs a KS4 patch
              and a re-sort. The rest of the layer is specified only

    The third slot was `link` (matching units across recordings) until
    2026-09-11, when concatenating consecutive recordings was measured to cost
    nothing at the join AND to remove the duplicates a matcher would have had
    to rediscover. Concatenate, do not stitch. The `link/` placeholder was
    removed on 2026-10-01.

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


def code_commit(repo=None, package='unitassay'):
    """The git commit this package is running from; '+dirty' ONLY if the
    PACKAGE ITSELF has uncommitted or untracked (non-ignored) changes; None
    outside a git checkout (an installed wheel).

    Stamped into every curated.json (2026-10-01): `version` alone stayed 0.1.0
    across every rule change since September, so no output could be traced to
    the code that produced it.

    SCOPED TO THE PACKAGE (2026-10-01). The first version asked about the WHOLE
    repository, so editing a figure script in paper/ during a run stamped 141
    of 147 sessions '+dirty' although not one package file had changed. A
    stamp must describe the code that ran, and paper/, tests/ and docs are not
    that code. Agrita: "there should not be dirty or ambiguous tag".
    """
    import subprocess
    from pathlib import Path
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parent.parent
    try:
        h = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=10)
        if h.returncode != 0:
            return None
        d = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain',
                            '--', package],
                           capture_output=True, text=True, timeout=10)
        return h.stdout.strip() + ('+dirty' if d.stdout.strip() else '')
    except Exception:
        return None


__author__ = 'Agrita Dubey, Qiao (Betty) Xu'
__license__ = 'MIT'

from . import curate, screen            # noqa: F401
