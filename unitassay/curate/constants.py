#!/usr/bin/env python3
"""curate/constants.py — thresholds that more than one module must agree on.

DEPENDENCY-FREE ON PURPOSE: `taxonomy.py` is pure Python (no numpy/sklearn) and
must stay importable in light contexts, while `bimodality.py` needs numpy,
sklearn and diptest. Neither can own a shared constant without dragging its
dependencies into the other, so shared constants live here and both import them.

WHY THIS FILE EXISTS AT ALL (2026-08-12). `LOW_SPIKE_THRESHOLD` was documented as
"derived: change min_group_spikes and this must follow" -- but `min_group_spikes`
was defined as a local `MIN_GROUP = 120` in **twelve separate analysis scripts**
and nowhere central, while taxonomy.py hardcoded `240`. So the stated dependency
was not real: changing one script's floor would silently leave 240 wrong. The
derivation is now expressed in code instead of in a comment.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""

# ── the L-ratio's per-group floor ────────────────────────────────────────────
# The mutual L-ratio inverts EACH group's OWN covariance in the multichannel PC
# space (d = 3 PCs x 10 channels = 30), which is degenerate at n ~ 40. 120 is
# ~3x that degeneracy point.
#
# HONEST STATUS: 120 is HAND-SET. The n~40 degeneracy is a real property of the
# statistic, but nothing calibrates why 3x rather than 2x or 4x. So everything
# derived from it is derived RELATIVE to a hand-set number -- justifying this one
# against a calibrated null is still open. Do not describe quantities
# below as "derived from first principles"; they are derived from THIS.
MIN_GROUP_SPIKES = 120

# ── the spike-count floor below which a cluster is not judged ────────────────
# DERIVED, not chosen: the overmerge test compares TWO groups, and each needs
# MIN_GROUP_SPIKES to support its own covariance, so a cluster cannot form a
# single comparison below 2 x MIN_GROUP_SPIKES. Measured resolvability on all
# 18,974 clusters confirms the floor is exactly there and is hard, not gradual:
#
#     0-120  120-200  200-240 | 240-400  400-700  700-1500  1500+
#      0.0%     0.0%     0.0% |   66.2%    96.2%     98.6%   99.8%
#
# Applied to the WHOLE-RECORDING spike count (`n_spikes`), absolute, no
# per-frame variant and no carve-outs (threshold raised from 200 to the derived
# floor).
LOW_SPIKE_THRESHOLD = 2 * MIN_GROUP_SPIKES        # = 240

# ── rule 5's time-domain gate ───────────────────────────────────────────────
# A THRESHOLD CURVE, NOT A NUMBER, and the gate x category matrix is what forced
# that. A single scalar (11.82, the 95th percentile of the whole far-pair null)
# looked calibrated overall and was miscalibrated everywhere: false-decline 8.3%
# below 500 spikes, 4.7% mid, 1.0% above 1500 -- over-rejecting small clusters
# and barely testing large ones. Same lesson as the re-detection null: ship the
# calibration PROCEDURE.
#
# DERIVED, not chosen: knots are the 95th percentile of `ccg_z_peak_trough` on
# 17,389 FAR PAIRS from the 147 sessions -- clusters sharing no channel, so two
# distinct neurons by construction -- taken per decile of min(n_a, n_b).
# Interpolated linearly in log n and CLAMPED at both ends. Verified on the null
# itself: 5.2% / 5.0% / 4.7% across the three size bands, 5.0% overall.
#
# NOT parametric. A log-linear fit had R^2 0.871 but extrapolated T towards zero
# where the measured curve flattens at ~3.4, which produced 16.8% false declines
# in the largest band -- worse than the scalar it replaced.
#
# lambda is NOT the right parameter: its 95th percentiles are non-monotonic
# (peaking near lambda~7), because `max` over 8 discrete Poisson bins is not
# standardised by lambda alone.
CCG_Z_CURVE_N = (279., 363., 462., 576., 731., 932., 1254., 1757., 2820., 5987.)

# ONE CURVE PER ACCUSER COUNT. Rule 5 tests EVERY accusing neighbour, so a
# cluster with k accusers gets k chances to exceed its threshold. At a per-PAIR
# 5% the per-CLUSTER false-decline rate would be 1-(1-0.05)^k -- ~10% at k=2 and
# ~19% at k=4, concentrated on the 31% of rule-5 clusters that have more than one
# accuser, and it would look like a real effect.
#
# So each pair is tested at the Sidak-corrected alpha_k = 1-(1-0.05)^(1/k), read
# off the SAME 17,389 far-pair null at a higher quantile. No new data, no new
# assumption beyond the correction itself.
#
# VERIFIED by resampling the null with k accusers per cluster: per-cluster
# false-decline 4.8 / 4.7 / 5.6 / 5.5% for k = 1..4 against a 5% target.
#
# Sidak assumes independence between a cluster's accusers. They share one member,
# so they are positively correlated and the correction is CONSERVATIVE -- it errs
# toward fewer false declines, which is the right direction for a rule that
# discards putative neurons.
CCG_Z_CURVE_T_BY_K = {
    1: (21.88, 19.49, 18.49, 14.89, 12.42, 9.27, 6.17, 4.41, 3.52, 3.43),
    2: (37.64, 31.48, 33.05, 27.18, 22.24, 19.01, 12.67, 8.83, 4.73, 4.72),
    3: (47.55, 40.59, 39.95, 33.45, 27.13, 23.15, 15.23, 12.20, 6.34, 5.98),
    4: (55.28, 49.45, 45.84, 44.99, 31.32, 25.47, 19.09, 16.04, 7.96, 6.68),
    5: (59.54, 53.84, 49.03, 49.34, 36.20, 31.72, 21.98, 19.15, 9.29, 8.70),
    6: (68.04, 58.85, 55.88, 65.17, 38.99, 34.82, 24.40, 21.81, 11.00, 10.03),
}
CCG_K_MAX = 6          # k>6 is clamped: only 14 clusters in 4,908 exceed it

# ── the DEFICIT side of the same window ─────────────────────────────────────
# The excess and the deficit detect DIFFERENT errors and must be separate tests:
#   EXCESS   coincidences above chance -> the same event detected twice
#            (re-detection). Handled by the z curve above.
#   DEFICIT  coincidences below chance -> one neuron split in two, whose own
#            refractory period forbids the halves co-firing (OVER-SPLIT).
#
# A single two-sided statistic cannot do both, and the shipped one silently did
# neither on the deficit side: measured on constructed partitions it fired at
# 4.3% against a 5% null -- the entire oversplit signal was being discarded,
# because counts cannot fall below zero so a deficit is bounded at z >= -sqrt(lam),
# about -2 at realistic lambda and far under any threshold.
#
# NO EMPIRICAL NULL IS NEEDED HERE, unlike the excess side. The excess statistic
# is a max over bins and has no closed form, so it was calibrated on 17,389 real
# far pairs. The deficit is the SUM over the central window, which under
# independence is exactly Poisson(lam) -- so the lower-tail probability is
# calibrated analytically. Verified on constructed far pairs: 4.1% fire at
# alpha=0.05, conservative because the Poisson is discrete at small lam.
#
# WHY NOT THE BARE `obs == 0` RULE, which catches 65.8% of partitions against
# 20.4% for this test. Partition pairs have median lam 1.9 where far pairs have
# 6.7 -- splitting one neuron halves both rates -- so most of that extra
# sensitivity is a RATE difference, not a refractory one. Conditioning on lam
# removes the confound. The rate signal may well be real, but it is not the
# refractory signature and must not be named as one.
CCG_DEFICIT_ALPHA = 0.05


def poisson_cdf(obs, lam):
    """P(X <= obs) for X ~ Poisson(lam). Small obs, so a direct sum is exact.

    Written out rather than imported from scipy: curate() is pure dict logic on
    the hot path (165 us/cluster) and should not pull in scipy for a sum of a
    dozen terms.
    """
    import math
    if lam is None or obs is None or lam <= 0:
        return None
    k = int(obs)
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(total, 1.0)


def ccg_deficit_alpha(k=1):
    """Sidak-corrected alpha for the deficit test over k accusers."""
    kk = int(max(1, min(k, CCG_K_MAX)))
    return 1.0 - (1.0 - CCG_DEFICIT_ALPHA) ** (1.0 / kk)


def ccg_z_threshold(n_min, k=1):
    """Declining threshold for one pair, corrected for the cluster's accuser count.

    ONE implementation, never inlined -- the curve is the calibration and a second
    copy would drift.

    ABOVE THE TOP KNOT THE CURVE IS FLAT, and that is an assumption rather than a
    measurement: the trend over the calibrated range falls steeply (21.88 -> 3.43),
    so clamping at 3.43 for n_min >= 5,987 extrapolates a declining curve as
    constant. The 147-session corpus sits inside the range (median 1,274), so this
    does not affect it -- but hybrid injected units carry ~68,000 spikes, ~11x
    beyond the top knot, and are scored on that extrapolation. Do not report a
    hybrid decline rate without saying so.
    """
    import bisect
    import math
    if n_min is None:
        return None
    kk = int(max(1, min(k, CCG_K_MAX)))
    ys = CCG_Z_CURVE_T_BY_K[kk]
    xs = CCG_Z_CURVE_N
    n = float(max(n_min, 1.0))
    if n <= xs[0]:
        return ys[0]
    if n >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, n) - 1
    f = (math.log(n) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
    return ys[i] + f * (ys[i + 1] - ys[i])

# ── the dip / L-ratio decision thresholds ───────────────────────────────────
# THESE TWO ARE ONE OPERATING POINT, NOT TWO THRESHOLDS. The rule is
# `dip_pval < DIP_PVAL_THRESH AND l_ratio < L_RATIO_THRESH`, and only the
# CONJUNCTION has a meaningful error rate. Measured on MEArec known single
# units, d=30, where any 'overmerged' verdict is a false positive by
# construction:
#
#      n     dip alone   L-ratio alone   BOTH (the rule in force)
#    240        10.3%          13.6%            4.7%
#    500         7.6%           5.9%            5.8%
#   1000         7.4%           5.9%            5.9%
#   2000         8.4%           9.4%            5.7%
#   4000         8.3%          14.8%            5.3%
#
# So each threshold ALONE is badly calibrated (6-15% type-1) while the pair
# lands at ~5% and is stable across a 17x range of n. Each statistic has a heavy
# left tail on single units, but the two tails do not coincide, so requiring
# both filters them. Do not quote, tune or interpret either number on its own.
#
# DO NOT try to calibrate them per statistic. A 5%-FP quantile threshold is
# DEGENERATE: dip would need 0.012 (n=240), 0.0023 (n=500) and 0.0 for
# n>=1000, and L-ratio 0.0 at every n -- i.e. the statistic could never fire.
# Replacing 0.05 with "the calibrated value" would destroy the test.
#
# AND DO NOT permute per cluster at runtime. Measured cost is ~320 ms/cluster
# for ONE round (phase1_bootstrap_pilot timing), so a 300-cluster session at
# 1,000 reps is ~27 hours. The null depends on (n, d) rather than on which
# cluster it is, so if a null is ever needed it is built ONCE offline over an
# (n, d) grid and shipped as a lookup -- runtime cost then is an interpolation.
#
# *** THE ~5% DOES NOT SURVIVE DRIFT (P1, measured 2026-08-12). ***
# Measured on a controlled factorial where drift is the
# ONLY thing that varies (same generation pipeline, well-recovered known single
# units, any fire = false positive):
#
#     drift        recursive pipeline FP     this conjunction FP
#     none (ctrl)          3.37%                   4.26%     <- matches the table above
#     slow                20.46%                  10.12%
#     fast                14.67%                  14.37%
#
# At n>=1000, where everything resolves: none 0-6.5%, slow 21.7-25.6%, fast
# 15.5-17.3%. So drift raises false positives 4-6x for the adopted pipeline and
# 2.4-3.4x for this conjunction. The control reproducing 4.26% is what makes the
# drift numbers credible.
#
# CONSEQUENCE: the 4.7-5.9% table above is conditional on DRIFT-FREE units and
# must never be quoted as the method's error rate. Real recordings drift -- this
# is a drift-analysis paper. Re-tuning is now a live requirement, not a
# formality, and what gets re-tuned is the CONJUNCTION, not either number.
#
# Slow drift is WORSE than fast (25.6% vs 17.3% at n=2000), which inverts the
# naive expectation and is not yet explained.
#
# BURSTING NOW MEASURED TOO (2026-08-13), on a
# regenerated build where it actually took effect (3-4x excess of ISIs <10 ms and
# higher amplitude CV in the burst arm; note n_isi_under_1ms is 0 in BOTH arms
# and is NOT a valid indicator -- a well-simulated burst cell does not violate a
# 1 ms refractory period):
#
#     drift    burst    noburst    burst effect
#     none     10.80%     4.69%      +6.11 pp
#     slow     27.45%    23.09%      +4.36 pp
#     fast     24.80%    21.69%      +3.12 pp
#
# So this conjunction's ~5% holds only for units that are BOTH drift-free AND
# burst-free (6.50% here). Bursting alone takes it to 14.90%. The two effects are
# SUB-ADDITIVE -- the burst effect shrinks as drift grows -- which is evidence
# they feed ONE mechanism (a unit's own variability becoming resolvable) rather
# than two independent ones. Also unmeasured: whether this drift effect
# operates on real clusters. `drift_category` in the records CANNOT test it --
# 17,124/18,974 are None, meaning uncategorised rather than drift-free, and the
# labelled categories are a curated subset. A real test needs per-cluster drift
# measured from spike_positions.npy.
#
# The Gaussian null once intended for this was refuted (1-3 orders of magnitude
# too narrow); the empirical MEArec null replaced it.
DIP_PVAL_THRESH = 0.05
L_RATIO_THRESH = 0.05

# Measured type-1 error of the CONJUNCTION above, by cluster size, on known
# single units. Quote this instead of "p < 0.05" when describing the rule.
CONJUNCTION_TYPE1_BY_N = {240: 0.047, 500: 0.058, 1000: 0.059,
                          2000: 0.057, 4000: 0.053}

# Minority component firing below this is not a candidate unit. RATE ONLY:
# the spike-fraction half was dropped because a fraction is
# contaminated by the other component's activity.
SATELLITE_RATE_HZ = 0.1

# ── drift, as a REPORTED risk rather than a corrected one ────────────────────
# drift_ratio = (range of per-time-bin median y) / (within-bin sigma), i.e. how
# far a cluster moves relative to its OWN spread. Measured per cluster from
# spike_positions.npy.
#
# WHY IT IS REPORTED AND NOT CORRECTED. Two attempts to remove the drift-driven
# false positives both failed, for the same structural reason:
#   * per-frame testing pays with n -- 6 frames cut spikes/frame ~6x, so slow
#     drift FP 33.3% -> 6.7% cost 2-unit detection 87.7% -> 36.8%
#   * detrending pays with SIGNAL -- regressing features on y position gives the
#     best FP (0.0% / 6.7%) and destroys 2-unit detection (87.7% -> 14.0%),
#     because in a genuine merge the two units sit at DIFFERENT DEPTHS, so
#     position variance IS the merge signal
# Drift and multi-unit structure are confounded in this feature space by
# construction. So the honest output is a quantified risk annotation.
#
# THE FLAG POINT IS PRINCIPLED, NOT TUNED: 1.0 is where drift EQUALS the
# cluster's own within-frame spread (1 sigma). Measured consequence on real
# clusters with n>=1000: 18.6% flagged, overmerge rate 24.0% above vs 11.1%
# below (2.2x).
#
# BUT THERE IS NO ELBOW -- the risk is a smooth monotone gradient, so the
# boolean is a coarse summary and `drift_ratio` itself must be carried per
# cluster and reported:
#     threshold  0.50   0.60   0.75   1.00   1.25   1.50   2.00
#     ratio      1.6x   1.8x   2.0x   2.2x   2.5x   2.8x   3.2x
DRIFT_RATIO_FLAG = 1.0

# ── the screen layer's L-ratio channel: the two numbers its criterion turns on
# A frame is CLEAN when EVERY comparison (noise + each neighbour) is below
# L_RATIO_MS_THRESH; the cluster PASSES when clean_frames / n_frames >=
# ISOLATED_T_RATIO. Mirrors unitassay/screen/config.py and
# unitassay/screen/curation/auto_curate.py.
# NOT "max across frames" -- that is strictly harsher and disagreed with
# the screen layer's own decision on 17.6% of the clusters it passed.
L_RATIO_MS_THRESH = 0.2
ISOLATED_T_RATIO = 0.6

# Above this share of drift frames with NO SPIKES, a cluster cannot reach
# ISOLATED_T_RATIO however clean its present frames are, so its failure is
# COVERAGE rather than contamination. DERIVED, not chosen: 1 - 0.6 = 0.4.
ABSENT_FRAME_FRAC_THRESH = 1.0 - ISOLATED_T_RATIO
