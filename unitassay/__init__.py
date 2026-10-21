"""UnitAssay: post-sorting curation for reliable single-unit identification.

UnitAssay is applied after spike sorting and provides complementary screening,
curation, and drift-analysis utilities.

    screen    evaluates cluster isolation and its stability over time using
              L-ratio-based comparisons against a local reference channel and
              neighbouring clusters

    curate    applies the sequential curation flow and records the evidence
              supporting each cluster-level decision

    drift     provides utilities for evaluating Kilosort drift estimates,
              including multi-bank gap-block diagnostics

The package was developed and validated with Neuropixels recordings processed
with CatGT and Kilosort 4. Benchmark results and the validated pipeline are
documented in the repository README and accompanying manuscript.

Example
-------
>>> from unitassay import curate
>>> rows = curate.curate_session(ks4_dir, isolation_dir="isolation")
>>> clean = [row for row in rows if row["outcome"] == "clean"]

Authors: Agrita Dubey and Qiao (Betty) Xu
License: MIT
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
