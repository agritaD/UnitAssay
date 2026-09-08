#!/usr/bin/env python3
"""curate/redetection.py

THE RE-DETECTION TEST: is this cluster a second copy of spikes another cluster
already holds?

WHY THIS IS A SEPARATE CHANNEL. The tool's other three tests -- LDL (within),
ISI-VR (temporal), mind_snag's L-ratio (between) -- all ask about a cluster's
FEATURES, or about its own spike train. None of them compares WHEN two clusters
fire. So none of them can see the case where two clusters hold the same physical
events, and on MEArec ground truth that case is 79% of everything we label
over-splitting: 50.5% of fragments share >70% of their spikes with a neighbour
and only 7.7% are genuinely disjoint.

Two mechanisms produce an extra cluster and only one of them is visible here:

  RE-DETECTION  KS4 fits a template, subtracts it, and re-fits the residual. If
                the first fit was imperfect the leftover still looks spike-like
                and a second template claims it, so both clusters contain the
                SAME event at a fixed non-zero lag. THIS TEST.
  PARTITION     one neuron's spikes divided between clusters (drift, amplitude
                change). Each spike is in exactly one cluster, so the trains are
                disjoint in time and no temporal test can ever see them. That is
                the between-cluster L-ratio's job, and the boundary is deliberate.

THE LAG MATTERS. KS4 aligns every template to its own peak, so per-cluster median
lag spans -2.3 to +1.9 ms and the same event appears at a fixed offset between two
clusters. Counting coincidences about ZERO reports ~13% overlap for clusters that
share most of their spikes. So the window follows each pair's own modal lag.

THE THRESHOLD IS NOT A CONSTANT. It is calibrated per recording against a null of
pairs that cannot be the same neuron: peak channels >= FAR_UM apart in PHYSICAL
distance with no shared channel above 10% of peak. That null needs no ground truth
and no verdict of ours. Two properties of it are load-bearing:

  * it is much higher on real data than on MEArec, whose spike trains are
    independent Poisson -- so a MEArec-derived number does not transfer;
  * it RISES as spike count falls, because the statistic maximises over 60
    candidate lags and N_PARTNERS partners and so saturates when there are few
    spikes to match. The threshold is therefore a CURVE over spike count.

CHANNEL INDEX IS NOT DISTANCE. Separation is computed from channel_positions.npy
in um. Real Neuropixels output happens to be depth-ordered so index arithmetic
would work there, but MEArec orders channels column-by-column, where index-98
pairs are 100 um apart -- which silently filled the null with genuine duplicates.

DEPENDS ON CatGT. CAR collapses this null's tail (session 90th percentile 41.7%
-> 5.7%) while leaving the median alone. On un-CAR'd data the band thresholds run
99/95/91/59% and the test is inert.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import numpy as np

# Physical separation beyond which two clusters cannot share a spike footprint.
# 400 um reproduces exactly what ">= 40 channels apart" gave on the real probe,
# and means the same thing on any geometry.
FAR_UM = 400.0
N_PARTNERS = 25          # candidate partners, both for the signal and the null
COINC_MS = 0.5           # half-width of the coincidence window about the modal lag
LAG_SEARCH_MS = 3.0      # range searched for the modal lag
LAG_BIN_MS = 0.1
MIN_SPIKES = 30          # below this no overlap statistic is computed at all
# Minimum |modal lag| for a coincidence to count as a re-detection. Two lag bins
# from zero, which is the smallest offset the 0.1 ms binning resolves, and still
# well inside the +-0.5 ms coincidence window. See is_redetection_pair.
LAG_MIN_MS = 0.2

# THE OVERLAP FLOOR IS A DEFINITION, NOT A CALIBRATION OUTPUT (Agrita, 2026-08-25).
# "Redundant" asserts that a substantial share of this cluster's spikes are already
# counted under another cluster. A calibrated quantile does not enforce that
# sentence: once the lag conjunction was applied the null's 98th percentile fell to
# ~5%, and at 5% the criterion removed 25.1% of real passes -- statistically valid
# at its stated false-positive rate, and plainly not a description of "a copy".
#
# So the floor is set by what the word means and the null REPORTS the resulting
# false-positive rate instead of choosing it. Swept over 5/10/20/30/50/70% on both
# corpora (m260825_redetection_floor_sweep.py), 20% is the best point on every
# real-data instrument at once:
#
# RAISED 20% -> 50% (Agrita, 2026-08-26), because 20% did not satisfy the
# sentence the category name asserts. A cluster 20% of whose spikes are shared is
# 80% NEW, and calling it "Redundant" overclaims in exactly the way a 5% floor
# would have. At >50% the majority of the cluster is already counted, which is the
# plain reading of the word and one defensible line in a methods section.
#
# 50% ALSO DOMINATES 70% on every composite measure, so this is not a compromise
# between the name and the numbers (m260826_floor_tradeoff.py, and note the two
# columns answer different questions -- see the module note on units of analysis):
#
#   floor  cluster acc  cluster F1  unique lost  dups left in  spike F1
#     20%       85.3%       68.5%    1,136,322       112,648     79.9%
#     50%       82.9%       65.3%      721,885       396,627     80.0%   <-- adopted
#     70%       81.0%       62.8%      467,174       857,053     74.5%
#
# 70% wins ONLY on unique spikes destroyed, and pays for it by leaving twice the
# duplication in the data.
#
# MEArec prefers 5% and is OVERRULED, deliberately: it contains no synchrony, so
# the low-overlap population it rewards us for deleting is one whose real-data
# confound the simulator cannot exhibit. Recall is flat across the whole sweep
# (70.4-71.8%), so the floor trades specificity, not neurons.
OVERLAP_FLOOR = 0.50
FOOTPRINT_FRAC = 0.10    # a channel is "in the footprint" above this share of peak

# Spike-count bands the null is stratified over. The first is below
# LOW_SPIKE_THRESHOLD and so is empty for judged clusters; it exists so that a
# caller feeding unjudged clusters gets a defined answer instead of a KeyError.
BANDS = [(30, 240), (240, 700), (700, 1500), (1500, 3000), (3000, 6000),
         (6000, 10 ** 9)]
BAND_LBL = ['30-240', '240-700', '700-1.5k', '1.5-3k', '3-6k', '>6k']

# Target per-cluster false-positive rate against the far-pair null.
DEFAULT_FP = 0.02
MIN_NULL_PER_BAND = 20   # below this a band borrows the pooled null


def band_of(n):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= n < hi:
            return i
    return len(BANDS) - 1


def overlap_and_lag(A, B):
    """(share of A's spikes that are the SAME EVENT as a B spike, modal lag ms).

    Counted within +-COINC_MS of the pair's own modal lag, not about zero.
    A and B are sorted spike times in ms.
    """
    if len(A) < MIN_SPIKES or len(B) < MIN_SPIKES:
        return 0.0, 0.0
    i = np.searchsorted(B, A)
    d = np.full(len(A), np.inf)
    for off in (-1, 0, 1):
        j = np.clip(i + off, 0, len(B) - 1)
        nd = A - B[j]
        d = np.where(np.abs(nd) < np.abs(d), nd, d)
    w = d[np.abs(d) <= LAG_SEARCH_MS]
    if len(w) < 10:
        return 0.0, 0.0
    h, e = np.histogram(w, bins=np.arange(-LAG_SEARCH_MS,
                                          LAG_SEARCH_MS + LAG_BIN_MS / 2,
                                          LAG_BIN_MS))
    lag = float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))
    return float(np.mean(np.abs(d - lag) <= COINC_MS)), lag


def overlap(A, B):
    """Overlap only, for callers that do not need the lag."""
    return overlap_and_lag(A, B)[0]


def is_redetection_pair(o, lag, thresh):
    """The criterion: enough shared events, AT A NON-ZERO LAG.

    THE LAG REQUIREMENT IS NOT COSMETIC -- it is what makes the criterion mean
    "re-detection" on real data rather than "these two fire together".

    A re-detection sits at a fixed NON-ZERO offset, because KS4 aligns each
    template to its own peak and the two templates peak at different samples.
    Coincidence at ZERO lag is a different phenomenon: neighbouring neurons
    driven by common input, or a residual shared artifact. Measured on the 50
    real sessions, among pairs exceeding the overlap threshold:

        near pairs (candidate duplicates)   27% at non-zero lag
        far pairs (>=400 um, cannot be one neuron)   2%

    So the non-zero-lag population is 13x enriched in near pairs and is not
    explained by the null -- but the zero-lag majority is indistinguishable from
    what clusters 400 um apart do, and deleting those would be deleting real
    neurons that happen to fire together. On MEArec the distinction barely
    matters (86% of flagged pairs are already at non-zero lag) because a
    simulator has neither artifact nor common input; that is exactly why this
    could not have been found on synthetic data.

    LAG_MIN_MS IS NOT A FREE PARAMETER. The null applies the SAME conjunction, so
    whatever cut is chosen, the false-positive rate is still fixed at the
    requested quantile of the far-pair distribution. The cut only trades
    sensitivity. Note this is what made an OVERLAP_FLOOR necessary: applying the
    conjunction dropped the calibrated threshold to ~5%, low enough that the
    criterion stopped meaning what its name says. See OVERLAP_FLOOR above.
    """
    # The floor is the concept's minimum; the calibrated band threshold still
    # binds when a recording is noisy enough to push it ABOVE the floor. Whichever
    # is stricter wins, so neither a quiet recording nor a filthy one can weaken
    # what "redundant" means.
    return o > max(thresh, OVERLAP_FLOOR) and abs(lag) > LAG_MIN_MS


def scan_session(templates, spike_times_ms, spike_clusters, channel_positions,
                 cluster_ids=None):
    """Overlap of every cluster against its near partners, plus its null sample.

    Signal and null partners are selected by the SAME rule -- nearest first -- so
    they are matched in spike count. They differ only in whether the partner is
    close enough to share a footprint. Selecting the null's partners any other way
    (e.g. largest-first) inflates it, because more partner spikes means more
    chance coincidence.

    Returns a list of per-cluster dicts:
      clu, n_spikes, amp, pairs [(partner, o_self_in_partner, o_partner_in_self,
      partner_n, partner_amp, modal_lag_ms)], o_far (None if the recording offers
      no far partner). o_far already has the lag conjunction applied, so it is
      the null OF THE CRITERION, not of the overlap alone.
    """
    amp = np.ptp(templates, axis=1)
    pk = amp.argmax(axis=1)
    mask = amp >= FOOTPRINT_FRAC * amp.max(axis=1)[:, None]
    n_t = len(templates)
    tm = {int(c): np.sort(spike_times_ms[spike_clusters == c])
          for c in np.unique(spike_clusters)}
    tm = {c: v for c, v in tm.items() if c < n_t and len(v) >= MIN_SPIKES}
    ids = sorted(tm) if cluster_ids is None else [c for c in cluster_ids if c in tm]

    pos = np.asarray(channel_positions, float)
    peak_xy = pos[pk[:n_t]]

    out = []
    for a in ids:
        A = tm[a]
        others = [b for b in tm if b != a]
        dist = {b: float(np.hypot(*(peak_xy[a] - peak_xy[b]))) for b in others}
        near = sorted([b for b in others if (mask[a] & mask[b]).any()],
                      key=lambda b: dist[b])[:N_PARTNERS]
        far = sorted([b for b in others if dist[b] >= FAR_UM
                      and not (mask[a] & mask[b]).any()],
                     key=lambda b: dist[b])[:N_PARTNERS]
        pairs = []
        for b in near:
            oab, lag = overlap_and_lag(A, tm[b])
            oba, _ = overlap_and_lag(tm[b], A)
            pairs.append((int(b), oab, oba, int(len(tm[b])),
                          float(amp[b].max()), lag))
        # THE NULL APPLIES THE SAME CONJUNCTION AS THE SIGNAL. A far pair that
        # coincides at zero lag could never trigger the criterion, so counting
        # its overlap in the null would inflate the threshold and cost
        # sensitivity for nothing. Matching the null to the criterion is the
        # whole reason the false-positive rate stays interpretable.
        fo = []
        for b in far:
            o, lag = overlap_and_lag(A, tm[b])
            fo.append(o if abs(lag) > LAG_MIN_MS else 0.0)
        o_far = max(fo, default=None)
        out.append(dict(clu=int(a), n_spikes=int(len(A)),
                        amp=float(amp[a].max()), pairs=pairs,
                        o_far=o_far, n_far=len(far)))
    return out


def band_thresholds(scan, fp=DEFAULT_FP):
    """Per-spike-count-band threshold at a stated false-positive rate.

    Bands with fewer than MIN_NULL_PER_BAND null observations borrow the pooled
    null rather than silently returning a quantile of nothing. Which bands did so
    is returned, because a borrowed threshold is an assumption and the gate x
    category matrix has to be able to count it.
    """
    by_band = {i: [] for i in range(len(BANDS))}
    for r in scan:
        if r['o_far'] is not None:
            by_band[band_of(r['n_spikes'])].append(r['o_far'])
    pooled = [x for v in by_band.values() for x in v]
    g = float(np.quantile(pooled, 1 - fp)) if pooled else 1.0
    thr, borrowed = {}, set()
    for i in range(len(BANDS)):
        if len(by_band[i]) >= MIN_NULL_PER_BAND:
            thr[i] = float(np.quantile(by_band[i], 1 - fp))
        else:
            thr[i] = g
            borrowed.add(i)
    return thr, borrowed, {i: len(v) for i, v in by_band.items()}


def redundancy_verdicts(scan, thr, is_overmerge=None):
    """Which clusters are copies, and of what.

    Decided PER FLAGGED PAIR, not per connected component. The requirement is
    that no flagged pair survives with both members intact -- a vertex cover --
    and dropping each pair's loser is the greedy cover. Grouping by connected
    component instead takes the transitive closure: A~B and B~C puts A with C
    even when they are not flagged, and keeping one member of the group discards
    a distinct cluster for nothing. Measured: the component form scores 45.6%
    recall / 84.4% accuracy WITH AN ORACLE picking the survivor, worse than
    pairwise with an ordinary rule.

    The loser is the more-contained member, EXCEPT when the containing partner is
    itself condemned as an over-merge -- it holds these spikes because it
    swallowed them, so it is the one to drop. Only over-merge status overrides
    containment. The general form of that rule ("never let a failed cluster
    delete a passing one") buys more recall but loses 7 points of specificity,
    because it also protects passing fragments.

    is_overmerge: callable(cluster_id) -> bool. Absent means the override is off.

    Returns {cluster_id: partner_id_it_is_a_copy_of or None}. Every scanned
    cluster gets a key, so "absent" can be distinguished from "tested, clean".
    """
    om = (lambda x: False) if is_overmerge is None else is_overmerge
    verdict = {r['clu']: None for r in scan}
    n_by = {r['clu']: r['n_spikes'] for r in scan}
    for r in scan:
        a, na = r['clu'], r['n_spikes']
        for b, oab, oba, nb, _ab, lag in r['pairs']:
            if not (is_redetection_pair(oab, lag, thr[band_of(na)])
                    or is_redetection_pair(oba, lag, thr[band_of(nb)])):
                continue
            ma, mb = bool(om(a)), bool(om(b))
            drop_a = (ma and not mb) if ma != mb else (oab > oba)
            if drop_a:
                verdict[a] = b
            elif b in verdict:
                verdict[b] = a
            n_by.setdefault(b, nb)
    return verdict
