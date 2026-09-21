#!/usr/bin/env python3
"""curate/bimodality.py

Standard name (2026-07-14): the "LDL test" (LDA-Dip-Lratio) -- use this name
in new code, figures, and docs instead of ad hoc phrasing ("LDA dip+L-ratio
test", "LDA-bimodal check", etc.). Verdict outcomes: LDL-bimodal (dip_pval <
DIP_PVAL_THRESH AND l_ratio < L_RATIO_THRESH -- two real, separated sources)
vs LDL-unimodal (either criterion fails to reject -- one source).

Canonical bimodality test, agreed 2026-06-24 (m260624_overmerging.py) and
2026-07-13 (extended to cover m260624_oversplitting.py's two-cluster case).
Previously copy-pasted with identical math (but no shared import) across
m260629_overmerging_examples.py, m260706_bimodal_isi_pass_v*.py,
m260707_mindsnag_*.py, m260708_*.py, scan_genuine_overmerge.py, and
m260624_oversplitting.py. Also hand-ported (necessarily, due to a package
boundary -- see auto_curate.py's _is_lda_bimodal()) into the dev mind_snag
copy at Drift_Figure_Code/mind_snag_dev_channel_assignment_v2/ -- that copy
must be kept in sync with this file if the math here ever changes.

Method: LDA finds the direction maximising separation between two spike
groups (in KS4 multichannel PC-feature space, N_PC PCs x n_local_channels);
Hartigan's dip test on that 1D projection checks for two modes; L-ratio
(screen/Mahalanobis formula) between the two groups checks whether
they're cleanly separated in the full feature space (not just along the
LDA axis).

Two entry points, one shared core (_dip_lratio_core):
  - lda_dip_lratio(feats)      : WITHIN one cluster -- k-means(2) splits its
                                  own spikes into two tentative sub-groups
                                  first, then runs the core. Used to ask
                                  "is this one cluster actually two neurons?"
                                  (overmerging).
  - dip_lratio_between(fa, fb) : BETWEEN two already-separate clusters --
                                  no internal splitting, fa/fb are each
                                  cluster's own spikes directly. Used to ask
                                  "are these two clusters actually one
                                  neuron?" (oversplitting).

Both signals matter on their own — checking dip_pval alone can miss real
structure the LDA projection doesn't align with (a k-means split that
looks visually separated in 2D PCA can still project to a smeared,
unimodal-looking 1D histogram); checking l_ratio alone can't distinguish
"two separated blobs" from "one blob with an outlier tail."

The two use cases READ the same (dip_pval, l_ratio) pair in OPPOSITE
directions, and this module deliberately does not force one verdict
function on both:
  overmerging  (lda_dip_lratio):  low dip_pval AND low l_ratio  -> bimodal
                                  and separated -> two neurons merged into
                                  one cluster -> use verdict() below.
  oversplitting (dip_lratio_between): high dip_pval AND high l_ratio ->
                                  unimodal and overlapping when the two
                                  clusters are combined -> same neuron
                                  split into two clusters -> caller applies
                                  its own (inverted) threshold logic; see
                                  m260624_oversplitting.py's AND-gate.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

import numpy as np
import scipy.spatial.distance as _spdist
from numpy.linalg import norm
from scipy.stats import chi2 as _chi2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import diptest
except ImportError as e:
    raise ImportError("curate/bimodality.py requires the 'diptest' package") from e

N_PC = 3
MIN_SPIKES = 50
DIP_PVAL_THRESH = 0.05
L_RATIO_THRESH = 0.05


def _lda_direction(fa, fb, d):
    """Regularized Fisher LDA direction separating two flat feature groups.

    Returns (w, reg) -- unit vector w and the regularization matrix used
    (callers reuse the same reg for the L-ratio covariance inverse, matching
    the original m260624_overmerging.py / m260624_oversplitting.py formulas
    exactly) -- or (None, reg) on failure.
    """
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    Sw = ((len(fa) - 1) * np.cov(fa.T) + (len(fb) - 1) * np.cov(fb.T)) / (len(fa) + len(fb) - 2)
    reg = max(1e-6, 1e-4 * np.trace(Sw) / d) * np.eye(d)
    try:
        w = np.linalg.solve(Sw + reg, mu_b - mu_a)
    except np.linalg.LinAlgError:
        return None, reg
    nrm = norm(w)
    if nrm == 0:
        return None, reg
    return w / nrm, reg


def _mutual_l_ratio(fa, fb, d, reg):
    """max(L(fb vs fa's ellipsoid), L(fa vs fb's ellipsoid)) -- screen/
    Schmitzer-Torbert & Redish (2004) formula, both directions, more
    conservative (harder-to-pass) direction wins. Returns np.nan on failure.
    """
    l_vals = []
    for pcs_self, pcs_other in [(fa, fb), (fb, fa)]:
        try:
            VI = np.linalg.inv(np.cov(pcs_self.T) + reg)
            mu_s = pcs_self.mean(0, keepdims=True)
            mah = _spdist.cdist(mu_s, pcs_other, 'mahalanobis', VI=VI)[0]
            l_vals.append(float(np.sum(1 - _chi2.cdf(mah ** 2, df=d)) / len(pcs_self)))
        except Exception:
            return np.nan
    return float(max(l_vals))


def dip_lratio_between(fa, fb, min_spikes=MIN_SPIKES, min_group_spikes=None):
    """Core test between two ALREADY-SEPARATE spike groups (e.g. two KS4
    clusters' own PC features, channel-aligned via align_channels() if they
    come from different clusters -- see curate/features.py).

    fa, fb: (n_a, d) / (n_b, d) flat feature matrices, same d.

    min_group_spikes: optional, opt-in stricter per-group floor (int). When
    None (default), behavior is byte-identical to the original: the floor is
    max(min_spikes // 4, 2). When set, the floor becomes
    max(min_spikes // 4, 2, min_group_spikes) instead. Added 2026-07-24 after
    null-calibration (m260724_between_cluster_null_calibration.py) showed the
    L-ratio component false-fires ~90% of the time at n_per_group=50, d=30
    (this project's standard 30-dim multichannel feature space) -- the
    false-positive rate collapses to 0% once each group has >=75 spikes, with
    100-150 recommended as a safety margin. Default is left untouched so
    existing callers (e.g. lda_dip_lratio(), the mind_snag-mirroring
    retrieval-gate path) see zero change unless they explicitly opt in.

    Returns dict: dip_pval, l_ratio, proj (LDA projection, fa's points then
    fb's points, concatenated -- order doesn't affect dip_pval, diptest
    sorts internally), w (unit LDA direction, for projecting other points
    into the same space if needed). NaN/None fields on failure.
    """
    fail = {'dip_pval': np.nan, 'l_ratio': np.nan, 'proj': None, 'w': None}
    floor = max(min_spikes // 4, 2)
    if min_group_spikes is not None:
        floor = max(floor, min_group_spikes)
    if fa.shape[0] < floor or fb.shape[0] < floor:
        return fail
    d = fa.shape[1]
    if d < 2 or fb.shape[1] != d:
        return fail

    w, reg = _lda_direction(fa, fb, d)
    if w is None:
        return fail

    proj = np.concatenate([fa @ w, fb @ w])
    _, dip_pval = diptest.diptest(proj)
    l_ratio = _mutual_l_ratio(fa, fb, d, reg)

    return {'dip_pval': float(dip_pval), 'l_ratio': l_ratio, 'proj': proj, 'w': w}


# ── PHASE 2: the universal dip verdict ───────────────────────────────────
#
# WHY THIS EXISTS. lda_dip_lratio_bestk() only ever runs the dip test when
# BIC picks k==2. For k==1 it returns 'unimodal' and for k>=3 'multiway',
# BOTH with dip_pval=NaN -- no test at all. On the 50-session set that is
# 14,403 of 17,290 judged clusters (83.3%), and the two untested branches are
# treated OPPOSITELY: unimodal -> clean (10,758), multiway -> auto-FAIL via
# curate() rule 1 (2,795). Agrita's decision (2026-08-10): close the gap with
# ONE mechanism rather than describe it with more categories.
#
# THE MECHANISM. Dip-test COMPONENT PAIRS, for whatever k the model gives:
#   k >= 3  -> every k(k-1)/2 pair, strongest evidence wins
#   k == 2  -> as before
#   k == 1  -> impose a 2-component fit and dip its axis anyway
# This is NOT the retired forced-KMeans(2) test: the split comes from a GMM in
# the same family as the rest of the pipeline, and the thresholds below are
# validated against ground truth rather than assumed.
#
# EVIDENCE BEHIND THE PARAMETER CHOICES (see the universal-dip-plan memory):
#  - cov_type='diag'. Phase 0b: `full` costs d+d(d+1)/2 = 495 params per
#    component at d=30, so BIC almost cannot afford k>=2 (median best_k=2,
#    measured median BIC(k=2)-BIC(k=1) = +2132). `tied` cannot select k at all
#    (5.6% pile up at the k=20 ceiling). `diag` costs 30 params/component and
#    turns over properly (0.3% at ceiling, mode k=2-4).
#  - max_k=8. Agrita's call, biologically motivated: >8 distinct neurons
#    merged into one KS4 cluster is not plausible. It also bounds the pair
#    family at 28, which matters because K is a POWER knob -- every extra pair
#    widens the multiple-comparison family. 14.8% of clusters want k>8 under
#    diag, and capping costs them a median 2.1% of their BIC range. Crucially
#    best_k is NOT a unit count: diag uses extra components to describe
#    non-Gaussian density shape (drift, bursting, spatial decay).
#  - thresholds UNCHANGED at 0.05/0.05. Phase 1 measured the existing verdict
#    on 19 MEArec ground-truth clean units (3,524 draws): false-positive rate
#    5.48% pooled, stable 4.7-5.9% across n. They are already calibrated; the
#    Gaussian-simulation suggestion that they were far too conservative was an
#    artefact of assuming Gaussian features (real nulls are 1-3 orders of
#    magnitude wider). Do NOT "fix" these numbers.
#
# NOT YET VALIDATED, and the reason a separate measurement must follow: the
# 5.48% above was for ONE imposed pair. Taking the best of up to 28 pairs is a
# MAXIMUM statistic and will inflate type-I error. n_pairs_tested is returned
# so that inflation can be measured and corrected empirically rather than by
# assuming Bonferroni (which over 28 tests would demand p<0.0018 and destroy
# power).
UNIVERSAL_COV = 'diag'
SATELLITE_RATE_HZ = 0.1


def universal_dip_verdict(feats, min_spikes=MIN_SPIKES, min_group_spikes=None,
                          cov_type=UNIVERSAL_COV, duration_sec=None,
                          satellite_rate_hz=SATELLITE_RATE_HZ,
                          dip_thresh=DIP_PVAL_THRESH, l_thresh=L_RATIO_THRESH):
    """A dip verdict for EVERY cluster: ONE imposed 2-component split.

    Replaces the untested branches of lda_dip_lratio_bestk(), which only ever
    ran the dip test for best_k==2 -- 14,403 of 17,290 judged clusters (83.3%)
    got no test, and the two untested branches were treated OPPOSITELY
    (unimodal -> clean, multiway -> auto-FAIL via curate() rule 1).

    THE DESIGN IS DELIBERATELY MINIMAL, and that is an empirical result rather
    than a preference. Validation on MEArec ground truth
    (m260812_universal_dip_validation.py) compared this against a best-k +
    all-component-pairs version:

                          imposed k=2      all-pairs (max_k=8)
      FP, known 1 units   0.00% (n=1283)   20.96%, and 93.4% at 7-28 pairs
      sensitivity         24/24 2-unit     6/6 (but via the 93% FP path)
      runtime             31 ms/cluster    ~124 ms

    All-pairs fails because `diag` uses extra components to describe
    NON-GAUSSIAN DENSITY SHAPE (drift smearing, burst amplitude variation,
    spatial decay) rather than to count units, so among 28 pairs there is
    almost always one pair of adjacent carved pieces of a single cloud that
    looks separated along its own LDA axis. Taking the best of them is a
    maximum statistic over nearly-guaranteed apparent structure. So: no BIC
    sweep, no pair enumeration, no max_k.

    Multi-unit sensitivity (m260812_multiunit_merge_sensitivity): 100% for
    k=2, 3 and 4 constructed merges at realistic channel overlap (d>=12).
    Apparent shortfalls at d=6-9 came from combining units that barely overlap
    spatially, which are not realistic overmerge candidates. k>=5 is untested.

    Thresholds are UNCHANGED at 0.05/0.05 and must stay so: the existing
    verdict measures 5.48% FP on ground-truth single units (Phase 1), i.e. it
    is already calibrated. A Gaussian-simulation suggestion that it was far
    too conservative was an artefact -- real nulls are 1-3 orders of magnitude
    wider than Gaussian.

    Returns dict:
      verdict   'overmerged' | 'not_overmerged' | 'ambiguous' | 'insufficient'
      basis     'full' | 'degenerate_split' | 'negligible_satellite'
                | 'lopsided_no_lratio' | 'none'
      dip_pval, l_ratio, proj, w, labels
      n_small, n_large, balance   (n_small / n_total)

    Three branches return 'not_overmerged' for different reasons, and `basis`
    keeps them distinguishable -- in particular `lopsided_no_lratio` is a known
    BLIND SPOT, not a clean pass (see below).

    `duration_sec` enables the satellite rate filter: a minority component
    firing below `satellite_rate_hz` is not a candidate unit. RATE-ONLY by
    Agrita's decision (2026-08-11) -- the old minority_frac<0.10 half was
    dropped because frac is contaminated by the OTHER component's activity,
    and the cases it uniquely governed had 27-spike satellites. Pass None to
    skip the filter, which is the conservative choice.

    KNOWN BLIND SPOT (`lopsided_no_lratio`). A satellite above
    satellite_rate_hz but holding fewer than `min_group_spikes` has no reliable
    adjudicator, so it conservatively returns not_overmerged. The 120-spike
    floor is an L-RATIO requirement (_mutual_l_ratio inverts EACH group's own
    covariance, degenerate at n~40 in d=30) while the dip would still run --
    but a dip-only verdict was measured at 18% FP and withdrawn: a lopsided
    split of a heavy-tailed unimodal cloud puts the minority in the TAIL, and
    the core-vs-tail axis genuinely shows a bump, so the dip reports the
    bimodality of a tail rather than of a neuron. Mitigating: genuine merges of
    comparable units give a roughly BALANCED split (median balance 0.495 for
    equal-size ground-truth pairs at d=30), so lopsided splits mostly arise
    from tail-carving of single units, where not_overmerged is correct anyway.
    """
    n, d = feats.shape
    base = {'verdict': 'insufficient', 'basis': 'none', 'dip_pval': np.nan,
            'l_ratio': None, 'proj': feats[:, 0] if d > 0 else np.zeros(n),
            'w': None, 'labels': np.zeros(n, dtype=int),
            'n_small': 0, 'n_large': n, 'balance': 0.0}
    if n < min_spikes or d < 2:
        return base

    try:
        gm = GaussianMixture(n_components=2, covariance_type=cov_type,
                             random_state=0, n_init=1).fit(feats)
        labels = gm.predict(feats)
    except Exception:
        return base

    fa, fb = feats[labels == 0], feats[labels == 1]
    n_small, n_large = min(len(fa), len(fb)), max(len(fa), len(fb))
    out = {**base, 'labels': labels, 'n_small': int(n_small),
           'n_large': int(n_large), 'balance': float(n_small / n)}

    # A split that COLLAPSES is evidence the cluster is not two separable
    # populations -- a result, not absence of evidence. (Previously returned
    # 'insufficient', leaving 6.6% of clusters with no verdict at all.)
    if n_small < 4:
        return {**out, 'verdict': 'not_overmerged', 'basis': 'degenerate_split'}

    if duration_sec is not None and duration_sec > 0:
        if (n_small / duration_sec) < satellite_rate_hz:
            return {**out, 'verdict': 'not_overmerged',
                    'basis': 'negligible_satellite'}

    if min_group_spikes is not None and n_small < min_group_spikes:
        return {**out, 'verdict': 'not_overmerged', 'basis': 'lopsided_no_lratio'}

    try:
        res = dip_lratio_between(fa, fb, min_spikes=min_spikes,
                                 min_group_spikes=min_group_spikes)
    except Exception:
        return out
    if res['w'] is None or np.isnan(res['dip_pval']) or np.isnan(res['l_ratio']):
        return out
    dp, lr = float(res['dip_pval']), float(res['l_ratio'])
    return {**out, 'verdict': verdict(dp, lr, dip_thresh=dip_thresh, l_thresh=l_thresh),
            'basis': 'full', 'dip_pval': dp, 'l_ratio': lr,
            'proj': feats @ res['w'], 'w': res['w']}


def pca2d(feats):
    """2D PCA projection for scatter-plotting the same feature matrix."""
    c = feats - feats.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return c @ Vt[:2].T


SEPARATION_STATES = ('separated', 'not_separated', 'conflicting', 'untested')

# WITHIN-cluster reading: two groups inside ONE cluster separating means the
# cluster holds more than one unit -- a FAILURE.
_WITHIN_NAMES = {'separated': 'overmerged', 'not_separated': 'not_overmerged',
                 'conflicting': 'ambiguous', 'untested': 'untested'}

# BETWEEN-cluster reading of the SAME result: two DIFFERENT clusters separating
# means they are genuinely distinct units -- a PASS. Same statistics, opposite
# consequence.
_BETWEEN_NAMES = {'separated': 'confirmed_separate',
                  'not_separated': 'indistinguishable',
                  'conflicting': 'conflicting', 'untested': 'untested'}


def separation_verdict(dip_pval, l_ratio, dip_thresh=DIP_PVAL_THRESH,
                       l_thresh=L_RATIO_THRESH):
    """DO THESE TWO GROUPS SEPARATE?  The domain-neutral primitive (#25).

    Returns one of SEPARATION_STATES and says NOTHING about whether separating
    is good or bad, because that depends on what the two groups are:

        within ONE cluster    -> separated means an OVERMERGE    (bad)
        between TWO clusters  -> separated means DISTINCT UNITS  (good)

    `verdict()` used to be the only entry point and returned the WITHIN-cluster
    words, so between-cluster callers were reading "overmerged" and mentally
    inverting it, or re-deriving the thresholds inline. One name with opposite
    meanings in two contexts is how a mislabelled figure gets made months later.
    Use this and name the consequence yourself, or use verdict() /
    between_verdict(), which name it for the two known contexts.
    """
    if np.isnan(dip_pval) or np.isnan(l_ratio):
        return 'untested'
    bimodal = dip_pval < dip_thresh
    separated = l_ratio < l_thresh
    if bimodal and separated:
        return 'separated'
    if not bimodal and not separated:
        return 'not_separated'
    return 'conflicting'


def verdict(dip_pval, l_ratio, dip_thresh=DIP_PVAL_THRESH, l_thresh=L_RATIO_THRESH):
    """WITHIN-cluster naming of separation_verdict(): 'overmerged' |
    'not_overmerged' | 'ambiguous' | 'untested'.

    Now a thin wrapper, so the thresholds live in exactly one place. Behaviour is
    unchanged -- the same mapping it always applied, just no longer the only way
    to ask the question.
    """
    return _WITHIN_NAMES[separation_verdict(dip_pval, l_ratio,
                                           dip_thresh=dip_thresh,
                                           l_thresh=l_thresh)]


def between_verdict(dip_pval, l_ratio, dip_thresh=DIP_PVAL_THRESH,
                    l_thresh=L_RATIO_THRESH):
    """BETWEEN-cluster naming: 'confirmed_separate' | 'indistinguishable' |
    'conflicting' | 'untested'.

    Exists so between-cluster callers stop importing `verdict` and reading
    "overmerged" to mean "these two clusters are distinct" -- the inversion #25
    is about.
    """
    return _BETWEEN_NAMES[separation_verdict(dip_pval, l_ratio,
                                             dip_thresh=dip_thresh,
                                             l_thresh=l_thresh)]


# ── KNOWN FAILURE MODE of universal_dip_verdict, measured 2026-08-12 ─────
#
# COLLINEAR multi-unit merges are MISSED ENTIRELY. Synthetic k-unit merges of
# 500-spike Gaussians in d=30, 30 reps per cell:
#
#   k    sep      collinear    random direction
#   2    4 sigma  0%           0%          <- sensitivity floor, both geometries
#   2    >=6      100%         100%
#   3,4  4/6/8    0%  (all 'ambiguous')    100%
#
# Mechanism: an imposed 2-way split of 3+ collinear components groups them as
# {1} vs {2} ALONG the line; the doubled group's inflated spread keeps the
# L-ratio above threshold while the dip still fires, so the halves disagree and
# the verdict is 'ambiguous'.
#
# WHY THIS MATTERS: collinear is the REALISTIC geometry on a linear probe --
# units differing mainly in depth on shared channels give feature offsets along
# one direction (the amplitude/position gradient). And 'ambiguous' is NOT a
# fail: curate() rule 7 defers it to screen_pass, so such a merge can PASS.
#
# CONSEQUENCE: do NOT use this to judge the `multiway` population (2,795
# clusters currently auto-failed by rule 1). Keep the auto-fail until the
# graded-separation multi-unit scenarios in
# docs/mearec_scenario_request_2026_08_12.md (P2) resolve it. The MEArec k=3
# result (100% at d>=12) rested on 4 non-collinear triples -- too thin to have
# caught this.
#
# The two candidate designs fail COMPLEMENTARILY, which is not visible from
# either measurement alone: all-pairs handles collinear triples correctly (it
# tests adjacent {0,1} and {1,2} pairs, each cleanly separated) but costs 93.4%
# false positives; imposed k=2 has 0.00% FP but misses collinear multiway. One
# direction worth testing: keep imposed k=2 as primary and escalate ONLY on an
# 'ambiguous' verdict with a non-lopsided split, which bounds the multiplicity
# cost to a small subset.


# ── RECURSIVE variant: fixes the collinear blind spot ─────────────────────
#
# DIAGNOSIS (m260812_collinear_blindspot_figure.py). In every failing case the
# DIP FIRES (p=0) and the L-RATIO blocks the verdict -- both limits are L-ratio
# limits, not dip limits. Mechanism: the L-ratio asks "does the other group's
# mass fall inside my ellipsoid?", and a group holding MORE THAN ONE unit has an
# inflated covariance whose ellipsoid engulfs its neighbour. Collinear k=3 can
# only be cut {2} vs {1}, so one side is always compound -> L-ratio 0.117.
#
# FIX: never compare a COMPOUND group. Split recursively until the children are
# atomic, then compare atomic groups. Same mechanism applied recursively, which
# keeps the "one universal test" property.
#
# THE GATE IS THE DIP ALONE, and that choice is forced: gating on the full
# verdict would stop immediately on collinear clusters (they return
# 'ambiguous'), i.e. on precisely the case being fixed. The dip fires on ~8-10%
# of real single-unit imposed splits (ground truth, Phase 1), so spurious
# recursion does happen -- it is caught at the end, because leaf pairs from a
# wrongly-split single unit are halves of one cloud and have a HIGH L-ratio.
# The residual risk is multiplicity from extra leaves, which is why MAX_DEPTH is
# small and the false-positive rate is measured rather than assumed.
RECURSE_MAX_DEPTH = 3


def _atomize(X, depth, min_spikes, min_group_spikes, cov_type, dip_thresh,
             max_depth, info=None):
    """Split while the DIP says there is internal structure. Returns leaves.

    The gate deliberately passes min_group_spikes=None to dip_lratio_between:
    the 120-spike floor is an L-RATIO requirement, and gating on it made a
    LOPSIDED sub-split stop the recursion, so the large side was never examined.
    The dip itself is valid at small n (1-D test on a pooled projection), so it
    can gate a split that the L-ratio could not adjudicate. The floor is still
    enforced later, where it belongs: leaves too small to support their own
    covariance simply do not participate in the final L-ratio comparisons.
    """
    # RECORD WHY WE STOPPED (#47). `atomic` used to conflate two very different
    # states: the dip ran and found no structure (a real negative result), and
    # the group could not be split at all (no dip ever computed). Both returned
    # a single leaf and both were reported `resolved`, so 111 clusters asserted an
    # adjudicated verdict on evidence that does not exist.
    def _stop(reason):
        if depth == 0 and info is not None:
            info.setdefault('atomic_reason', reason)
        return [X]

    if depth >= max_depth or X.shape[0] < 8:
        return _stop('too_small_or_deep')
    try:
        gm = GaussianMixture(n_components=2, covariance_type=cov_type,
                             random_state=0, n_init=1).fit(X)
        lab = gm.predict(X)
    except Exception:
        return _stop('gmm_failed')
    fa, fb = X[lab == 0], X[lab == 1]
    if min(len(fa), len(fb)) < 4:
        return _stop('degenerate_split')
    res = dip_lratio_between(fa, fb, min_spikes=min_spikes, min_group_spikes=None)
    if res['w'] is None or np.isnan(res['dip_pval']) or res['dip_pval'] >= dip_thresh:
        # ATOMIC -- and the dip value that established it is EVIDENCE, not
        # absence of evidence. It used to be discarded: with no leaf pair the
        # caller's dip_pval stayed NaN, so 4,587 `+large_atomic` clusters
        # printed "dip p = n/a" despite the dip having run and answered
        # "no internal structure" (Agrita, 2026-08-13: "unless the spike count
        # is less than 240 on main unit we should have a dip test result").
        if depth == 0 and info is not None:
            info['root_dip'] = (None if res['w'] is None
                                or not np.isfinite(res['dip_pval'])
                                else float(res['dip_pval']))
            info.setdefault('atomic_reason',
                            'dip_ran_no_structure' if info['root_dip'] is not None
                            else 'dip_not_computable')
        return [X]                     # no internal structure -> atomic
    # The dip that DID split is the evidence behind an `unadjudicable` verdict
    # (leaves exist, so structure was found -- only the L-ratio could not
    # adjudicate any pair). It used to be discarded because the recovery below
    # is gated on `atomic`, leaving 96 clusters reporting no dip at all.
    if depth == 0 and info is not None:
        info['split_dip'] = float(res['dip_pval'])
    return (_atomize(fa, depth + 1, min_spikes, min_group_spikes, cov_type,
                     dip_thresh, max_depth) +
            _atomize(fb, depth + 1, min_spikes, min_group_spikes, cov_type,
                     dip_thresh, max_depth))


def _best_leaf_pair(leaves, min_spikes, min_group_spikes):
    """Strongest separation evidence over leaf pairs that can support an L-ratio."""
    best = None
    floor = max(min_group_spikes or 0, 4)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            fa, fb = leaves[i], leaves[j]
            if min(len(fa), len(fb)) < floor:
                continue
            r = dip_lratio_between(fa, fb, min_spikes=min_spikes,
                                   min_group_spikes=min_group_spikes)
            if r['w'] is None or np.isnan(r['dip_pval']) or np.isnan(r['l_ratio']):
                continue
            key = (float(r['l_ratio']), float(r['dip_pval']))
            if best is None or key < best[0]:
                best = (key, float(r['dip_pval']), float(r['l_ratio']), r['w'], (i, j))
    return best


RESOLVED_BASES = ('full', 'recursive', 'large_atomic', 'large_tested',
                  'lopsided_no_lratio+large_atomic', 'lopsided_no_lratio+large_tested',
                  'degenerate_split+large_atomic', 'degenerate_split+large_tested')

# `unsplittable` (#47): a single leaf produced WITHOUT a dip -- the group
# could not be split at all, so there is no negative result, only absence.
UNRESOLVED_SUFFIXES = ('remainder_too_small', 'unadjudicable',
                      'unsplittable')


def is_resolved(basis):
    """Did a verdict actually come from running the test?  SINGLE SOURCE.

    Callers must NOT default `resolved` themselves.  `v.get('resolved', True)`
    was the bug: the `remainder_too_small` and `none` branches never set the
    key, so 12.2% of real clusters reported as adjudicated when the test could
    not run on them at all.  Same shape as the L-ratio mislabel -- an absent
    field read as a positive result -- so it gets the same treatment: one
    function, derived from `basis`, which is always recorded.
    """
    if basis is None or basis == 'none' or basis == '':
        return False
    if any(basis.endswith(sfx) for sfx in UNRESOLVED_SUFFIXES):
        return False
    return True


def overmerge_cleared(v, duration_s=None, max_satellite_hz=0.1):
    """Is a cluster CLEARED of overmerge -- strictly enough to act on?

    Use this, not `v['verdict'] == 'not_overmerged'`, wherever a clean verdict
    grants something (a burst rescue, a pass, a retrieval). Three conditions,
    and the third is the one that keeps being forgotten:

      1. the verdict is 'not_overmerged'
      2. it was actually computed  -- is_resolved(basis)
      3. nothing NON-NEGLIGIBLE was parked without being adjudicated

    (3) exists because measured on real data it is the common case, not an edge
    case. Of the 11 burst rescues the adopted test would newly allow, **9 park
    10-31% of the cluster's spikes (up to 0.65 Hz) unexamined** -- and a parked
    group is an unanswered question, not evidence of a single unit. Acting on
    those 9 would convert Fail -> Pass on clusters that may well hold a second
    unit in the parked group. Only 2 of the 11 are defensible.
    See m260812_burst_gate_test_swap.py.

    duration_s is required to judge negligibility as a RATE (the adopted
    criterion, Agrita 2026-08-11: rate only, the spike-fraction criterion was
    dropped). Without it this cannot clear a verdict that parked anything --
    it returns False rather than guessing.
    """
    if v is None or v.get('verdict') != 'not_overmerged':
        return False
    if not is_resolved(v.get('basis')):
        return False
    parked = v.get('parked') or 0
    if parked == 0:
        return True
    if not duration_s:
        return False                      # cannot judge a rate without a duration
    return (parked / float(duration_s)) < max_satellite_hz


def overmerge_verdict(feats, min_spikes=MIN_SPIKES, min_group_spikes=None,
                          cov_type=UNIVERSAL_COV, duration_sec=None,
                          satellite_rate_hz=SATELLITE_RATE_HZ,
                          dip_thresh=DIP_PVAL_THRESH, l_thresh=L_RATIO_THRESH,
                          max_depth=RECURSE_MAX_DEPTH):
    """universal_dip_verdict, comparing ATOMIC groups instead of compound ones.

    Root cause it addresses: **the L-ratio is structurally a TWO-group
    statistic.** It fits ONE Gaussian to group A and asks how much of B falls
    inside that ellipsoid, so a group holding two units is silently mis-modelled
    as one wide blob and the ellipsoid engulfs its neighbour. mind_snag never
    hits this because it compares against each KS4 neighbour SEPARATELY -- its
    groups are exogenous and presumed atomic. Ours are fitted, so pooling is
    possible, and the fix is to never compare a compound group.

    Measured: synthetic collinear k=3 and k=4 go 0% -> 100% detection at
    4/6/8 sigma; false positives on MEArec ground-truth single units stay flat
    (2.15% -> 2.19%).

    PARKED SMALL GROUPS ARE NOT A VERDICT (fixed 2026-08-12, Agrita).
    The first split can leave a group that cannot be adjudicated -- collapsed
    (<4 spikes), a sub-0.1 Hz fragment, or under the L-ratio's 120-spike floor.
    An earlier version returned `not_overmerged` for all of those, which meant
    **31.5% of real clusters got an untested pass** (lopsided 23.2%, degenerate
    8.2%) and silently defaulted 58% of the `multiway` population. Dismissing a
    satellite says nothing about whether the remaining ~90% is a merge, so now
    the small group is PARKED and the LARGE group is tested. `basis` records
    both facts, e.g. 'lopsided_no_lratio+large_tested'.

    Only `insufficient` (n < min_spikes) is a true dead end.

    Returns universal_dip_verdict's keys plus n_leaves, leaf_sizes, parked.
    """
    n, d = feats.shape
    first = universal_dip_verdict(feats, min_spikes, min_group_spikes, cov_type,
                                  duration_sec, satellite_rate_hz, dip_thresh, l_thresh)
    if first['basis'] == 'none':
        return {**first, 'n_leaves': 0, 'leaf_sizes': [], 'parked': None,
                'tested': False, 'resolved': False}

    # If the first split parked an unadjudicable group, keep testing the REST.
    work, parked = feats, None
    tested_mask = np.ones(n, dtype=bool)     # every row, until something is parked
    if first['basis'] in ('degenerate_split', 'negligible_satellite',
                          'lopsided_no_lratio'):
        lab0 = first['labels']
        g0, g1 = feats[lab0 == 0], feats[lab0 == 1]
        big, small = (g0, g1) if len(g0) >= len(g1) else (g1, g0)
        if big.shape[0] < max(min_group_spikes or 0, 4) * 2:
            # Even the remainder cannot be tested: the L-ratio needs each group
            # to support its own covariance, so two groups require
            # 2*min_group_spikes spikes (240 by default). resolved=False is
            # essential -- it was defaulting to True, which made a cluster the
            # test COULD NOT EXAMINE look adjudicated. Measured consequence:
            # every ground-truth draw at n<=200 landed here, so the apparent
            # 0.00% false-positive rate at small n was the test never firing,
            # not the test succeeding (m260812_lowspike_reliability.py).
            return {**first, 'basis': first['basis'] + '+remainder_too_small',
                    'n_leaves': 1, 'leaf_sizes': [int(big.shape[0])],
                    'parked': int(small.shape[0]),
                    'tested': False, 'resolved': False}
        # Keep the MASK, not just the subset: `proj` must come back aligned to
        # the caller's rows. Returning a full-length `labels` beside a
        # subset-length `proj` is a latent broadcast bug for any consumer using
        # both -- the diagnostic panels hit it as (331,) vs (314,).
        big_is_0 = len(g0) >= len(g1)
        tested_mask = (lab0 == 0) if big_is_0 else (lab0 == 1)
        work, parked = big, int(small.shape[0])

    atom_info = {}
    leaves = _atomize(work, 0, min_spikes, min_group_spikes, cov_type,
                      dip_thresh, max_depth, info=atom_info)
    sizes = [int(len(L)) for L in leaves]
    suffix = '' if parked is None else '+large_tested'
    out = {**first, 'n_leaves': len(leaves), 'leaf_sizes': sizes, 'parked': parked}

    best = _best_leaf_pair(leaves, min_spikes, min_group_spikes)
    if best is None:
        # Distinguish the two reasons -- conflating them is the same mistake one
        # level down. On real data 327/335 of these are (a): a REAL negative
        # result, not a default. Only 8/1227 clusters (0.7%) are (b).
        #   (a) ONE leaf  -> the group was tested and found ATOMIC
        #   (b) >1 leaves -> leaves exist but none can support an L-ratio
        atomic = len(leaves) <= 1
        stem = first['basis'] if parked is None else first['basis']
        tail = ('+atomic' if atomic else '+unadjudicable') if parked is None else \
               ('+large_atomic' if atomic else '+large_unadjudicable')
        # Report the atomicity dip when the first split exited early and so
        # never recorded one. Without this the verdict says "not_overmerged"
        # with no visible evidence, which reads as untested.
        dp_atomic = out.get('dip_pval')
        if dp_atomic is None or not np.isfinite(dp_atomic):
            # atomic -> the dip that established atomicity; not atomic ->
            # the dip that established there IS structure (#47). Previously
            # only the first was recovered.
            dp_atomic = (atom_info.get('root_dip', dp_atomic) if atomic
                         else atom_info.get('split_dip', dp_atomic))
        # A single leaf is only a RESULT if a dip actually ran. When the group
        # could not be split (too small, GMM failure, degenerate split) there is
        # no evidence, and calling it resolved is the absent-is-not-negative
        # error one level down.
        if atomic and atom_info.get('atomic_reason') not in (
                None, 'dip_ran_no_structure'):
            tail = tail.replace('atomic', 'unsplittable')
        return {**out, 'verdict': 'not_overmerged', 'basis': stem + tail,
                'dip_pval': dp_atomic,
                'tested': True, 'resolved': bool(atomic)}
    _, dp, lr, w, pair = best
    v = verdict(dp, lr, dip_thresh=dip_thresh, l_thresh=l_thresh)
    # input-aligned projection: NaN marks rows parked before the comparison, so
    # a caller can plot it against `labels`/its own features without reindexing.
    proj_full = np.full(n, np.nan, dtype=float)
    proj_full[tested_mask] = work @ w
    return {**out, 'dip_pval': dp, 'l_ratio': lr, 'w': w, 'proj': proj_full,
            'tested_mask': tested_mask,
            'leaf_pair': pair, 'verdict': v, 'tested': True, 'resolved': True,
            'basis': ('recursive' if parked is None
                      else first['basis'] + suffix)}
