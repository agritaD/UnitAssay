"""CLI for layer 3. Lives here rather than in gap_blocks so that `python -m`
does not re-execute a module the package __init__ has already imported.

    python -m unitassay.drift <ks4_dir> [more dirs...]
"""
import sys

from unitassay.drift.gap_blocks import _main

if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
