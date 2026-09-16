"""Detect KS4's gap blocks, and EXCLUDE them from the session drift.

This does NOT fix the bug. It drops the contaminated measurements. The
distinction matters and the names here keep it: `exclude_gap_blocks`, not
`correct`.

THE BUG. Kilosort 4 splits the probe into `2 * nblocks - 1` half-overlapping
drift blocks by slicing the DEPTH BIN INDEX range (`datashift.py`, the `ifirst`
construction). Nothing in that slicing consults the channel map. On a multi-bank
Neuropixels recording the banks are separated by a stretch of probe carrying no
recording sites, so one or more blocks land squarely in that dead gap.

A block with no electrodes in it still gets a `dshift` value. That value is
interpolation plus noise, not measurement: it tracks the mean of its two
neighbours at r = 0.742 while running 1.26x above it.

IT DOES NOT ALWAYS READ HIGH. Across the 51 real gap sessions the correction
moves the session value DOWN in 37 and UP in 14. The bug injects an unanchored
interpolate-plus-noise term, and that term usually but not always exceeds the
true local drift. `inflation` is named for the median case; read it as a ratio,
not as a guarantee of direction.

MEASURED AGAINST KNOWN TRUTH. On synthetic recordings where the probe is
verifiably stationary — true drift exactly zero — gap blocks report 1.67 um
against 0.33 um for populated blocks of the SAME recordings. That is 5.0x
inflation of a quantity whose true value is nought. On 147 real sessions the
same paired, within-session contrast is 9.46 um against 6.88 um (Wilcoxon
p = 0.002), and 72 of the 147 have a bank gap.

EXCLUSION IS NOT A FIX, AND THIS IS THE WHOLE CAVEAT. What the code does is
identify the affected blocks from the geometry and drop them from the session
statistic. The drift over the gap then ends up NOT MEASURED rather than measured
correctly — defensible, since there are no electrodes there to measure from, but
it is not a repair.

It cannot be a repair: recovering what those blocks SHOULD have reported would
need the drift fingerprint they were estimated from, and KS4 does not retain it.

**Actually fixing bug 5 means patching KS4** — make the `ifirst` block split in
`datashift.py` channel-map aware so blocks are laid over populated depth ranges —
and then re-sorting. That has not been done.

    from unitassay.drift import gap_blocks
    rep = gap_blocks.exclude_gap_blocks('/path/to/ks4_output')
    print(rep['range_90_gap_excluded'], rep['gap_block_ratio'])

Run directly for a one-line report per sort:

    python -m unitassay.drift /path/to/ks4_output [more dirs...]
"""
from pathlib import Path

import numpy as np

from unitassay.drift.metrics import range_90_per_block

# A block holding under this fraction of the mean per-block channel count is a
# gap block. Read from the geometry, never assumed from the bank count.
EMPTY_FRAC = 0.25
# Below this y-span a probe is single-bank and cannot have a gap.
SINGLE_BANK_SPAN_UM = 4500.0

__all__ = ['detect', 'exclude_gap_blocks', 'EMPTY_FRAC']


def _load(ks_dir):
    ks = Path(ks_dir)
    ops_f = ks / 'ops.npy'
    if not ops_f.exists():
        raise FileNotFoundError(f'no ops.npy in {ks}')
    ops = np.load(ops_f, allow_pickle=True).item()
    d = ops.get('dshift')
    if d is None:
        raise ValueError(f'{ks} has no dshift: drift correction was disabled '
                         f'(nblocks=0), so there are no blocks to check')
    d = np.asarray(d, dtype=float)
    d = d.reshape(d.shape[0], -1)
    yblk = np.asarray(ops['yblk'], dtype=float).ravel()
    if yblk.size != d.shape[1]:
        raise ValueError(f'{ks}: {yblk.size} block centres for {d.shape[1]} '
                         f'dshift columns')
    pos_f = ks / 'channel_positions.npy'
    if not pos_f.exists():
        raise FileNotFoundError(f'no channel_positions.npy in {ks}; the check '
                                f'is geometric and cannot run without it')
    y = np.load(pos_f)[:, 1].astype(float)
    return d, yblk, y


def detect(ks_dir):
    """Which drift blocks sit in a channel gap.

    Returns per-block channel counts, the boolean gap mask, and the geometry it
    was decided from. Never raises on a single-bank probe — it simply finds no
    gap blocks, which is the correct answer.
    """
    d, yblk, y = _load(ks_dir)
    span = float(y.max() - y.min())
    # Blocks half-overlap, so a block's extent is one block spacing either side
    # of its centre.
    step = float(np.median(np.diff(yblk))) if yblk.size > 1 else span
    n_in = np.array([int(np.sum(np.abs(y - c) <= step)) for c in yblk])
    thresh = EMPTY_FRAC * float(n_in.mean())
    is_gap = n_in < thresh
    return dict(
        ks_dir=str(ks_dir), n_blocks=int(d.shape[1]), n_batches=int(d.shape[0]),
        y_span_um=span, block_spacing_um=step,
        banks=1 if span < SINGLE_BANK_SPAN_UM else 2,
        channels_per_block=n_in.tolist(), gap_threshold_channels=thresh,
        is_gap_block=is_gap.tolist(), n_gap_blocks=int(is_gap.sum()),
        gap_block_idx=np.where(is_gap)[0].tolist())


def exclude_gap_blocks(ks_dir):
    """Session drift computed over the POPULATED blocks only.

    `range_90_gap_excluded` is the number to report; `range_90_all_blocks` is
    what KS4's own blocks give. `gap_block_ratio` is all_blocks /
    gap_excluded — a RATIO, not a direction: it runs below 1 on 14 of the 51
    real gap sessions.
    """
    rep = detect(ks_dir)
    d, _, _ = _load(ks_dir)
    per_block = range_90_per_block(d)
    is_gap = np.asarray(rep['is_gap_block'], dtype=bool)

    all_mean = float(per_block.mean())
    if is_gap.all():
        raise ValueError(f'{ks_dir}: every block is a gap block; the geometry '
                         f'or the block count is not what this check assumes')
    pop_mean = float(per_block[~is_gap].mean())
    gap_mean = float(per_block[is_gap].mean()) if is_gap.any() else float('nan')

    rep.update(
        range_90_per_block=per_block.tolist(),
        range_90_all_blocks=all_mean,
        range_90_gap_excluded=pop_mean,
        range_90_gap_blocks=gap_mean,
        gap_block_ratio=all_mean / pop_mean if pop_mean else float('nan'),
        worst_populated_block=float(per_block[~is_gap].max()),
        gap_blocks_excluded=bool(is_gap.any()),
        method='gap blocks dropped from the session mean — NOT a repair',
        note=('Exclusion, not repair. The drift fingerprint those blocks were '
              'estimated from is not retained, so what they SHOULD have read '
              'cannot be recovered. Fixing bug 5 itself means patching KS4 '
              "(datashift.py `ifirst`) and re-sorting."),
    )
    return rep


def _label(p):
    """KS4 output dirs are all named alike, so show enough path to tell them
    apart."""
    q = Path(p)
    lead = q.parts[-4] if len(q.parts) >= 4 else ''
    lab = f'{lead}/{q.name}' if lead else q.name
    return lab if len(lab) <= 38 else lab[:37] + '…'


def _main(argv):
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print('\nusage: python -m unitassay.drift.gap_blocks <ks4_dir> [...]')
        return 2
    worst = 0
    for a in argv:
        try:
            r = exclude_gap_blocks(a)
        except (FileNotFoundError, ValueError) as e:
            print(f'{_label(a):<40} SKIPPED — {e}')
            worst = max(worst, 1)
            continue
        if r['n_gap_blocks']:
            word = 'inflated' if r['gap_block_ratio'] >= 1 else 'DEFLATED'
            print(f"{_label(a):<40} {r['banks']}-bank  "
                  f"{r['n_gap_blocks']} GAP block(s) at {r['gap_block_idx']}  "
                  f"range_90 {r['range_90_all_blocks']:.2f} -> "
                  f"{r['range_90_gap_excluded']:.2f} um  "
                  f"({word} {r['gap_block_ratio']:.2f}x)")
        else:
            print(f"{_label(a):<40} {r['banks']}-bank  no gap blocks  "
                  f"range_90 {r['range_90_gap_excluded']:.2f} um")
    return worst


if __name__ == '__main__':
    import sys
    sys.exit(_main(sys.argv[1:]))
