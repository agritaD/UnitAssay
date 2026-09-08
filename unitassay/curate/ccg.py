#!/usr/bin/env python3
"""curate/ccg.py

Cross-correlogram, factored out of m260708_neighbor_fail_examples.py (the
oversplitting diagnostic: refractory dip at t=0 -> same neuron split in two;
flat -> genuinely independent sources). Same use case applies to checking
whether a dip-test's k-means(2) split is a real second neuron or an
artificial split of one clean unit.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

import numpy as np


def compute_ccg(t_a, t_b, window_ms=50.0, bin_ms=0.5, fs=30000.0):
    """Cross-correlogram: lags of t_b spikes relative to each t_a spike.

    t_a, t_b: spike times in RAW SAMPLES (divided by fs internally).
    Returns (bin_centers_ms, counts).
    """
    t_a_s = np.sort(t_a.astype(np.float64)) / fs
    t_b_s = np.sort(t_b.astype(np.float64)) / fs
    win = window_ms / 1000.0
    n_bins = int(round(2 * window_ms / bin_ms))
    edges = np.linspace(-window_ms, window_ms, n_bins + 1)
    lags_ms = []
    for ta in t_a_s:
        lo = np.searchsorted(t_b_s, ta - win)
        hi = np.searchsorted(t_b_s, ta + win)
        lags_ms.extend((t_b_s[lo:hi] - ta) * 1000.0)
    lags_ms = np.array(lags_ms)
    lags_ms = lags_ms[np.abs(lags_ms) > 1e-6]
    counts, _ = np.histogram(lags_ms, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def dip_verdict(centers, counts, refrac_ms=1.0):
    """Refractory-dip-ratio verdict for a CCG between two putative sub-populations.

    dip_ratio = mean count within +-refrac_ms / mean count in the 5-20ms flanks.
    Low ratio (<0.3) -> refractory dip -> likely one neuron split in two.
    High ratio (>=0.7) -> flat -> likely two independent sources.
    """
    win_mask = np.abs(centers) <= refrac_ms
    out_mask = (np.abs(centers) > 5) & (np.abs(centers) <= 20)
    mean_refrac = counts[win_mask].mean() if win_mask.any() else 0
    mean_outer = counts[out_mask].mean() if out_mask.any() else 1
    dip_ratio = mean_refrac / max(mean_outer, 1)
    if dip_ratio < 0.3:
        label = 'refractory dip -> likely same neuron (split artifact)'
    elif dip_ratio < 0.7:
        label = 'partial dip -> ambiguous'
    else:
        label = 'flat -> likely independent sources'
    return float(dip_ratio), label


# Rule 5's time-domain gate. ONE definition of the accuser statistic, because it
# was written out twice -- once in the synthetic run, once for the real sessions
# -- and the two are compared to the SAME threshold curve. A window or flank
# edit in one copy would have silently shifted only half the population.
CCG_WINDOW_MS = 25.0     # half-width of the correlogram
CCG_BIN_MS = 0.5
CCG_CENTRE_MS = 2.0      # |lag| <= this is the coincidence peak under test
CCG_FLANK_MS = (5.0, 20.0)   # the local Poisson rate is estimated here


def accuser_stats(t_a, t_b, fs=30000.0):
    """CCG evidence for ONE accusing neighbour, or None if it cannot be measured.

    Returns {z, n_min, obs, lam}:
      z      the peak-or-trough Poisson z over the central bins -- the EXCESS
             test's statistic, compared against ccg_z_threshold(n_min, k)
      n_min  spikes in the smaller train; the threshold is a curve in it
      obs    total observed counts in the central bins
      lam    counts expected there from the flank rate -- obs vs lam is the
             DEFICIT test (a refractory hole means one neuron, not two)

    None means the evidence is ABSENT, not clean: too few spikes, an empty
    centre or flank, or a non-positive flank rate. Callers must not substitute a
    default -- an accuser that could not be tested has not been cleared.
    """
    if t_a is None or t_b is None or len(t_a) < 2 or len(t_b) < 2:
        return None
    c, n = compute_ccg(t_a, t_b, window_ms=CCG_WINDOW_MS, bin_ms=CCG_BIN_MS, fs=fs)
    cen = np.abs(c) <= CCG_CENTRE_MS
    flk = (np.abs(c) > CCG_FLANK_MS[0]) & (np.abs(c) <= CCG_FLANK_MS[1])
    if not cen.any() or not flk.any():
        return None
    per_bin = float(n[flk].mean())
    if per_bin <= 0:
        return None
    zb = (n[cen] - per_bin) / np.sqrt(per_bin)
    return dict(z=float(max(np.max(zb), -np.min(zb))),
                n_min=int(min(len(t_a), len(t_b))),
                obs=float(n[cen].sum()),
                lam=per_bin * int(cen.sum()))
