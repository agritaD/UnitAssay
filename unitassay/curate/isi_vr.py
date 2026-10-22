#!/usr/bin/env python3
"""curate/isi_vr.py — THE screen layer's ISI-VR criterion. One implementation.

WHY THIS FILE EXISTS
--------------------
The screen layer's ISI-VR criterion had been reimplemented independently in
several measurement harnesses, and the reimplementation was WRONG in a way that
only shows up at large spike counts, so nothing on our corpora exposed it.

The screen layer's real rule (`unitassay/screen/curation/auto_curate.py`) is
TWO branches:

    num_viol < 5          -> pass. A small-n FLOOR: at low firing rates the Hill
                             denominator is tiny and a couple of stray
                             coincidences would blow the normalised rate past
                             threshold.
    num_viol >= 5         -> compute the HILL RATIO, observed / Poisson-expected,
                             and fail only if it reaches `isi_thresh` (0.5).

Every reimplementation kept the floor and dropped the ratio, collapsing the
second branch to an unconditional FAIL:

    passes_isi = not (len(viol) >= 5) or burst_like        # WRONG

That is a rate criterion turned into an absolute count. Measured on the 4,016
synthetic clusters it wrongly fails **223 (5.6%)**, every disagreement in the
same direction, and the error is purely size-dependent — ZERO below 1,000
spikes, 119 in 3,000-6,000, 101 above 6,000. The corpus median is 1,242 spikes,
which is why it stayed invisible.

**It surfaced on Buccino's hybrid corpus**, where injected units carry
54,000-85,000 spikes: 5 violations out of 70,000 is 0.007%, so the count form
failed 8 of 9 ground-truth units and `curate()` scored 11.1%
sensitivity. Under the real rule their Hill ratios are 9.14e-04 to 0.186 — all
far below 0.5 — so every one passes. The fix ("make it a rate, keep >=5 as the
small-n floor") is exactly what the screen layer already does; the defect was in
the reimplementations, not the criterion.

The burst override is deliberately NOT included here. It needs the multichannel
PC features and an LDA bimodality test, so it belongs with the caller that has
them; `is_burst` is threaded separately. Callers that had `or burst_like` keep
it, unchanged.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
import numpy as np

REFRACTORY_MS = 1.0     # the screen layer's REFRACTORY_PERIOD, in ms
ISI_THRESH = 0.5        # the screen layer's isi_thresh on the Hill ratio
NUM_VIOL_GUARD = 5      # the small-n floor -- a FLOOR, never a threshold


def violation_stats(spike_times_ms, refractory_ms=REFRACTORY_MS):
    """(n_viol, hill_ratio, n_spikes) for one cluster.

    `hill_ratio` is observed violations over the number a Poisson process of the
    same rate and duration would produce, using the cluster's OWN span (as
    auto_curate does — `T = st[-1] - st[0]`, not the recording length, so a unit
    present in only part of the session is not penalised for its absence).

    Returns hill_ratio = 0.0 when it cannot be formed (fewer than 2 spikes, or
    zero span). That matches auto_curate, and it is the PASSING value — which is
    safe here only because the guard below independently requires >= 5
    violations before the ratio is ever consulted.
    """
    t = np.sort(np.asarray(spike_times_ms, dtype=float))
    n = len(t)
    if n < 2:
        return 0, 0.0, n
    span = t[-1] - t[0]
    isis = np.diff(t)
    n_viol = int((isis < refractory_ms).sum())
    if span <= 0:
        return n_viol, 0.0, n
    lam = n / span                                   # spikes per ms
    expected = 2.0 * refractory_ms * lam * n
    return n_viol, (n_viol / expected if expected > 0 else 0.0), n


def passes_isi_vr(spike_times_ms, refractory_ms=REFRACTORY_MS,
                  thresh=ISI_THRESH, guard=NUM_VIOL_GUARD):
    """The screen layer's criterion, WITHOUT the burst override (see module docstring).

    True = the cluster's refractory violations are consistent with a single
    unit. The caller adds `or is_burst` if it has the burst evidence.
    """
    n_viol, ratio, _ = violation_stats(spike_times_ms, refractory_ms)
    if n_viol < guard:
        return True                                  # small-n floor
    return ratio < thresh
