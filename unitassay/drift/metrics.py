"""Drift metrics for a Kilosort sort: how much the probe moved, and how.

ONE NUMBER CANNOT DESCRIBE DRIFT. A steady ramp, a single step, a fast wobble and
pure estimator noise can all produce the SAME `range_90` while being physically
different and mattering differently. So three are reported:

    range_90   p95 - p5 of a block's trace.  The headline, and the one the
               Kilosort paper uses -- but see the trace-length caveat below.
    slow       peak-to-peak of the 60 s moving average.  THE ONE TO REPORT: it
               is what the sorter's interpolation actually corrects, and it is
               the component that drives over-merge.
    fast       RMS of the residual after that average.  Estimator jitter and
               real fast motion, mixed and not separable. Diagnostic only.
    max_step   largest change between consecutive batches. Catches the discrete
               jump that `slow` cannot tell from a ramp.

MATLAB's `prctile`, NOT numpy's `percentile`. MATLAB assigns sorted values the
plotting positions (i-0.5)/n; numpy's default uses (i-1)/(n-1). They differ by up
to a whole rank AT THE TAILS, which is exactly where p5 and p95 live. `_prctile`
here is the MATLAB convention and every percentile in this package goes through
it. Differences are ~0.65% at the median and up to 6% on short recordings, with
a systematic sign -- small, but not noise.

RANGE_90 GROWS WITH TRACE LENGTH. It is a spread statistic over the whole trace,
so a longer recording contributes more batches and a wider p5-p95 spread at an
identical drift rate. Comparing a 10-minute session against a 50-minute one
therefore compares sampling, not drift. Measured on 147 real sessions: an
apparent multi-bank increase with duration (9.4 -> 13.3 um) disappears entirely
when the window is held fixed (9.7 -> 9.1, p = 0.80). Use `slow`, or hold the
window constant, or compare within a session -- never between sessions of
different length.

THE NINE-BLOCK REQUIREMENT IS LOAD-BEARING for the artifact criteria, not
defensive. They measure COHERENCE ACROSS BLOCKS, so a rigid (nblocks=1) estimate
has nothing to be coherent with and is not scoreable. `session_metrics` returns
NaNs rather than a number in that case.

GAP BLOCKS ARE NOT EXCLUDED HERE. `range_90` and `session_metrics` take whatever
blocks they are given. On a multi-bank probe some of those blocks sit in the dead
gap between banks and report interpolation rather than measurement, reading 5.0x
high where the true drift is verified zero -- see `gap_blocks`. Exclude them
first with `gap_blocks.exclude_gap_blocks()` if the number is going into a
result.
"""
from pathlib import Path

import numpy as np
from scipy.stats.mstats import mquantiles

# Artifact-criteria thresholds, verbatim from the MATLAB
# `compute_artifact_flags.m` this was ported from.
STEP_Z_THRESH = 5.0      # a step of > 5 MADs is flagged
COH_THRESH = 0.6         # >= 60% of blocks stepping together = a global step
N_MIN = 2                # distinct probes on a day to trigger the day rule
WINDOW_FRAC = 0.3        # first/last 30% for the settling ratio
SETTLING_THRESH = 0.4
MIN_R90 = 15.0           # minimum range_90 for criteria 1 and 3
N_BLOCKS = 9             # the block count the criteria are defined for
SMOOTH_S = 60.0          # the slow/fast split, in seconds
BATCH_S = 2.0            # KS4 batch at the default 60000 samples / 30 kHz

__all__ = ['load_dshift', 'range_90', 'range_90_per_block', 'slow_fast',
           'max_step', 'session_metrics', 'artifact_flags',
           'STEP_Z_THRESH', 'COH_THRESH', 'N_MIN', 'WINDOW_FRAC',
           'SETTLING_THRESH', 'MIN_R90', 'N_BLOCKS', 'SMOOTH_S', 'BATCH_S']


def _prctile(a, p, axis=0):
    """MATLAB `prctile`. See the module docstring for why numpy's differs."""
    a = np.asarray(a, dtype=float)
    return np.apply_along_axis(
        lambda v: np.asarray(mquantiles(v, prob=np.asarray(p) / 100.0,
                                        alphap=0.5, betap=0.5)), axis, a)


def load_dshift(ks_dir):
    """Kilosort's drift estimate, (n_batches, n_blocks) in microns, or None."""
    f = Path(ks_dir) / 'ops.npy'
    if not f.exists():
        return None
    ops = np.load(f, allow_pickle=True).item()
    d = ops.get('dshift')
    if d is None:
        return None
    d = np.asarray(d, dtype=float)
    return d.reshape(d.shape[0], -1) if d.ndim else None


def range_90_per_block(dshift):
    """p95 - p5 for each block, MATLAB convention. Shape (n_blocks,)."""
    d = np.asarray(dshift, dtype=float)
    hi = _prctile(d, 95, axis=0).ravel()
    lo = _prctile(d, 5, axis=0).ravel()
    return hi - lo


def range_90(dshift):
    """MEAN over blocks of (p95 - p5). NaN if absent.

    Mean, not median: a median discards the non-rigid component that
    multi-block estimation exists to measure. On gap-free sessions the worst
    block runs a median 1.75x the median block. Report the worst block alongside.
    """
    if dshift is None or np.asarray(dshift).size == 0:
        return np.nan
    return float(np.mean(range_90_per_block(dshift)))


def slow_fast(dshift, batch_s=BATCH_S, smooth_s=SMOOTH_S, blocks=None):
    """Split each block's trace into slow (correctable) and fast (jitter).

    `slow` is the peak-to-peak of a moving average; `fast` is the RMS of what
    the average leaves behind. Edges are trimmed by half the kernel, because
    a same-mode convolution tapers there and would invent a step.
    """
    d = np.asarray(dshift, dtype=float)
    k = max(3, int(round(smooth_s / batch_s)))
    cols = range(d.shape[1]) if blocks is None else blocks
    slow, fast = [], []
    for j in cols:
        v = d[:, j]
        v = v[np.isfinite(v)]
        if v.size <= k + 2:
            continue
        sm = np.convolve(v, np.ones(k) / k, mode='same')
        e = max(k // 2 + 1, 1)
        sm_i, v_i = sm[e:-e], v[e:-e]
        slow.append(float(np.ptp(sm_i)))
        fast.append(float(np.sqrt(np.mean((v_i - sm_i) ** 2))))
    if not slow:
        return dict(slow=np.nan, fast=np.nan, frac_slow=np.nan, n_blocks=0)
    s, f = float(np.mean(slow)), float(np.mean(fast))
    return dict(slow=s, fast=f, frac_slow=s / (s + f) if (s + f) else np.nan,
                n_blocks=len(slow), slow_per_block=slow, fast_per_block=fast)


def max_step(dshift):
    """Largest change between consecutive batches, over all blocks."""
    d = np.asarray(dshift, dtype=float)
    if d.size == 0 or d.shape[0] < 2:
        return np.nan
    return float(np.nanmax(np.abs(np.diff(d, axis=0))))


def session_metrics(dshift):
    """range90, coh_frac, n_flagged, settling_ratio for one session.

    NaNs when the block count is not the 9 the artifact criteria are defined
    for, matching the MATLAB `continue`.
    """
    nan = dict(range90=np.nan, coh_frac=np.nan, n_flagged=np.nan,
               settling_ratio=np.nan)
    if dshift is None or np.asarray(dshift).shape[1] != N_BLOCKS:
        return nan
    d = np.asarray(dshift, dtype=float)
    T = d.shape[0]
    r90 = range_90(d)

    # criterion 1 input: a step in MADs of the pooled step distribution
    steps = np.abs(np.diff(d, axis=0))
    mad_step = float(np.median(steps)) + 1e-10
    step_z = steps / mad_step
    flag_idx = np.where(np.any(step_z > STEP_Z_THRESH, axis=1))[0]
    n_flagged = int(flag_idx.size)
    coh = (float(np.mean(np.sum(step_z[flag_idx] > STEP_Z_THRESH, axis=1))) / N_BLOCKS
           if n_flagged else 0.0)

    # criterion 3 input: does the drift settle? first vs last 30%
    w = max(5, int(np.floor(T * WINDOW_FRAC)))
    return dict(range90=r90, coh_frac=coh, n_flagged=n_flagged,
                settling_ratio=range_90(d[-w:]) / (range_90(d[:w]) + 1e-6))


def artifact_flags(sessions):
    """The three artifact criteria over a list of sessions.

    `sessions`: dicts with `ks_dir`, `date` and `np` (probe number). Returns the
    same list with metrics and flags added, so the caller keeps its own keys.
    """
    out = [dict(s, **session_metrics(load_dshift(s['ks_dir']))) for s in sessions]
    r90 = np.array([o['range90'] for o in out], dtype=float)
    coh = np.array([o['coh_frac'] for o in out], dtype=float)
    nfl = np.array([o['n_flagged'] for o in out], dtype=float)
    setr = np.array([o['settling_ratio'] for o in out], dtype=float)

    with np.errstate(invalid='ignore'):
        # C1 requires real drift: a mechanical jerk does not cause small drift
        is_fast = (coh >= COH_THRESH) & (nfl > 0) & (r90 > MIN_R90)

        # C2, day level: independent probes cannot develop one artifact by
        # chance, so a day with >= N_MIN flagged probes taints its whole day --
        # but only sessions that themselves show meaningful drift.
        dates = np.array([o['date'] for o in out])
        nps = np.array([o['np'] for o in out])
        is_day = np.zeros(len(out), dtype=bool)
        for d_ in np.unique(dates):
            hit = (dates == d_) & is_fast
            if np.unique(nps[hit]).size >= N_MIN:
                is_day[(dates == d_) & (r90 > MIN_R90)] = True

        is_global = is_fast | is_day
        is_step = (~is_global) & (setr <= SETTLING_THRESH) & (r90 > MIN_R90)

    for o, a, b, c, e in zip(out, is_fast, is_day, is_global, is_step):
        o.update(is_fast_coherent=bool(a), is_day_flagged=bool(b),
                 is_global=bool(c), is_step_disp=bool(e))
    return out
