#!/usr/bin/env python3
"""curate/taxonomy.py

Presentation-layer taxonomy for curate()'s output: collapses the rule-table
tag vocabulary (curate/decision_flow.py) into a flat category scheme agreed
with Agrita 2026-08-07, extended with a fourth outcome 2026-08-25.

    Pass  |  Fail  |  Low-spike-count  |  Redundant

`Redundant` is not a kind of failure -- see the comment on REDUNDANT below.

Each category holds mutually-exclusive SUB-TYPES. Every sub-type additionally
carries two INDEPENDENT annotations (`locus`, `mechanism`) and there are three
orthogonal overlay flags (`burst_rescued`, `needs_inspection`,
`high_drift`).

WHY ANNOTATIONS AND NOT NESTING (option 3, Agrita 2026-08-07)
--------------------------------------------------------------
Two descriptions of a failure were both wanted, and they cut across each
other rather than nesting:

    mechanism 'contamination' = { noise (WITHIN), neighbour-boundary
                                  (BETWEEN), neighbour-indistinguishable
                                  (BETWEEN) }
    locus 'within'            = { overmerge, isi_vr, noise contamination,
                                  untested, ambiguous }

Neither is a subset of the other, so making either one the nesting level
would split the other apart. Instead there is ONE row per sub-type and both
axes ride along as columns. The sub-type column is the one that sums.

`locus` follows Agrita's 2026-07-24 reorg rule, which is also how the figure
directories are already laid out (m260723_catgt_devcopy_category_examples.py
:43-51):
  within   -- the LDL feature-space test on the cluster's own spikes.
              INCLUDES noise contamination: the noise-channel L-ratio
              compares against THIS cluster's own reference channel, not a
              different cluster ("noise is something within cluster").
  temporal -- the ISI-VR spike-train test. Split out of 'within'
              2026-08-07, see LOCI below.
  between  -- a genuine comparison against a DIFFERENT cluster
              (dip_lratio_between).
  all      -- every test passed, so no single test is the locus ('clean').
  open     -- unattributed. `unexplained` was defaulted to between in July
              and explicitly flagged open in that same comment; option 3
              lets it say so instead of defaulting.
  cross-temporal -- two clusters' SPIKE TIMES compared. Neither 'between'
              (two clusters' feature clouds) nor 'temporal' (one cluster's own
              train), so it gets its own value rather than overloading either.
  n/a      -- not judged at all (below_threshold).

WHY A SEPARATE MODULE, NOT A RENAME IN decision_flow.py
-------------------------------------------------------
curate()'s tags name the RULE THAT FIRED; this module names WHAT THAT MEANS
for a reader of the paper. The rule table is validated against real data and
should not churn every time a display label is reworded, and its tag strings
are baked into the catgt_devcopy_*.json files already on disk and into ~8
analysis scripts. curate() stays the source of truth; this is a mapping.

THE ORGANISING PRINCIPLE: the rule that ISSUED the decision
------------------------------------------------------------
Sub-types partition their category because each corresponds to exactly one
terminal branch of curate() (or, for the neighbour split below, one cell of
dip_lratio_between's 2x2). That is what makes the columns sum.

WHY burst_rescued IS AN OVERLAY AND NOT A SUB-TYPE
---------------------------------------------------
Burst-rescue fires at rule 0, UPSTREAM of the rules that issue a PASS
(2/3/5), so a cluster can need both: 248 of the 492 burst-rescued PASS
clusters were ALSO shape_preserving_decay, before rule 3 was retired. As sibling
rows they would have double-counted and the Pass column would not have summed. `is_burst` in the records IS
already exactly the rescue flag -- compute_passes_isi_independent()
(m260723_catgt_devcopy_retrieval_gate.py) sets it only inside the
`if not passes_isi:` branch, and only when the burst gate flips the verdict.

WHY THE TOP-LEVEL LABEL IS 'Fail' AND NOT 'Confirmed Fail'
-----------------------------------------------------------
Clusters in it carry needs_inspection=True (pending/unexplained rows), so
"Confirmed" would assert a confidence they do not have. 'Pass' kept
symmetric.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

PASS = 'Pass'
FAIL = 'Fail'
LOW_SPIKE = 'Low-spike-count'
# FOURTH TOP-LEVEL OUTCOME (Agrita, 2026-08-24). A re-detection is not a BAD
# cluster -- its spikes are real and correctly sorted, they are simply already
# counted under another cluster. Filing it as Fail would put it in the same
# bucket as contaminated and over-merged units, which is a category error: the
# user's question "is this junk?" and "is this a copy of #47?" have different
# answers and different remedies (discard vs de-duplicate). Keeping it separate
# also keeps the denominators honest, since a redundant cluster should not count
# against specificity the way genuine noise does.
REDUNDANT = 'Redundant'
CATEGORIES = (PASS, FAIL, LOW_SPIKE, REDUNDANT)

# Whole-recording spike count below which a cluster is not judged at all.
# ABSOLUTE (Agrita, 2026-08-07): no rule overrides it, not even a rule that
# fired on real evidence. The sub-threshold isi_vr_fail clusters therefore
# move to Low-spike-count too -- accepted deliberately, because "units below
# the floor are not judged" is one sentence in a methods section and the
# carve-out needed a paragraph to justify 0.4% of the population.
#
# 240 IS DERIVED, NOT HAND-SET (2026-08-12). This replaced 200 once the dip
# test was run on ALL 18,974 clusters with no exclusion
# (m260812_full_population_dip.py):
#
#   The overmerge test compares TWO groups, and its L-ratio step inverts each
#   group's OWN covariance. A group therefore needs min_group_spikes to exist
#   at all, so a cluster needs 2*min_group_spikes = 2*120 = 240 before any
#   comparison can be formed. Measured resolvability confirms the floor is
#   exactly there and is hard, not gradual:
#
#       0-120  120-200  200-240 | 240-400  400-700  700-1500  1500+
#        0.0%     0.0%     0.0% |   66.2%    96.2%     98.6%   99.8%
#
#   Below 240 the test cannot run, so it returns a verdict that was never
#   computed. The old 200 sat BELOW the floor and so admitted the 420
#   clusters at 200-240 as judged when they can never be adjudicated -- the
#   same untested-default failure this codebase keeps hitting (see
#   curate.bimodality.is_resolved and lratio_state below).
#
# So the constant is now a consequence of min_group_spikes rather than an
# independent knob: change min_group_spikes and this must follow.
#
# The earlier empirical argument still CORROBORATES it, which is why 200 was
# close: on the real 50-session set the L-ratio's frame-to-frame CV
# (std_lr/mean_lr) falls 0.50 (50-100 spikes) -> 0.31 (100-150) -> 0.22
# (150-200) then flattens (0.24 at 200-300, 0.17 at 300-500), driven by
# n_valid_frames -- low-count clusters yield 4-5 drift frames instead of 6-7.
# And the judged Pass rate saturates over the same range: 46.8% (no
# threshold) -> 48.0% (150) -> 48.1% (200) -> 48.0% (250). Two independent
# lines, one structural and one empirical, landing in the same place.
#
# NOW validated against ground truth, which the old note said was impossible.
# The sim_hybrid set indeed has no low-count units, but MEArec clean units
# can be SUBSAMPLED to any size (m260812_lowspike_reliability.py): false
# positives are 0.00% from n=240 to n=1000. The surprise is at the OTHER end
# -- FP rises to 4.87% (1500) and 5.26% (2000-4000), because the dip's power
# grows with n until it resolves a single unit's drift tails and burst
# spread. This threshold protects the small end; the large end needs a
# caveat of its own that no spike-count floor can supply.
# Imported, not restated: the relationship IS the justification, so it must be
# executable. Previously this line read `LOW_SPIKE_THRESHOLD = 240` while
# min_group_spikes lived as a local `MIN_GROUP = 120` in twelve analysis
# scripts, so "change one and the other must follow" was not actually true.
# constants.py is dependency-free precisely so this module can stay pure Python
# (no numpy/sklearn) while sharing the number with bimodality.py.
from unitassay.curate.constants import (LOW_SPIKE_THRESHOLD, MIN_GROUP_SPIKES,  # noqa: E402
                             DRIFT_RATIO_FLAG, ISOLATED_T_RATIO,
                             ABSENT_FRAME_FRAC_THRESH)

assert LOW_SPIKE_THRESHOLD == 2 * MIN_GROUP_SPIKES, (
    'LOW_SPIKE_THRESHOLD is DERIVED as 2 * MIN_GROUP_SPIKES (two groups, each '
    'needing enough spikes to invert its own covariance). If you intend to '
    'change the floor, change MIN_GROUP_SPIKES in curate/constants.py.')

# dip_lratio_between's own thresholds -- MUST stay in sync with
# m260729_between_cluster_real_check.py:62, which is what produced the
# dip_pval/l_ratio values this module re-reads.
DIP_THRESH, L_THRESH = 0.05, 0.05

# --- annotation vocabularies -------------------------------------------
# 'temporal' split out from 'within' 2026-08-07, when the decision flow was
# restated as THREE co-equal tests (LDL / ISI-VR / L-ratio). ISI-VR is
# within-cluster in the loose sense that it uses only the cluster's own
# spikes, but it is a fundamentally different kind of evidence from the LDL
# feature-space test -- a spike-train property, not a cluster-shape one.
# Giving it its own locus makes the three tests map one-to-one onto three
# loci and three mechanisms, so the diagram and the table say the same
# thing:
#     LDL      within    -> overmerge
#     ISI-VR   temporal  -> refractory
#     L-ratio  between   -> contamination
# 'all' = every test passed, so no single test is the locus (clean only).
#
# A FOURTH ROW, 2026-08-24. The re-detection test compares two clusters' SPIKE
# TIMES, which is neither of the existing axes on its own: 'between' is about two
# clusters' feature clouds, 'temporal' is about one cluster's own train. Rather
# than overload 'between' -- which would break the documented 1:1 mapping of
# locus to test, and with it the correspondence between the table and the
# decision-flow figure -- it gets its own locus naming the composition:
#     overlap  cross-temporal -> redundancy
LOCI = ('within', 'temporal', 'between', 'cross-temporal', 'all', 'open', 'n/a')
# A MECHANISM NAMES A TEST, NEVER AN ERROR CLASS (Agrita, 2026-09-03: "make it
# mechanism only"). Four of these used to name the error the test was believed
# to prove -- 'overmerge', 'oversplit', 'drift', 'redundancy' -- which is the
# same overclaim the sub-type names carried, one field down. Measured, the worst
# was `overmerge`: the sub-type asserts OVER-MERGE and ground truth agrees 19.2%
# of the time. The error class now lives in curate/findings.outcome, which frees
# this field to say only what was measured.
MECHANISMS = ('bimodality', 'refractory', 'contamination', 'absence',
              'coincidence', 'time_ordering', 'overlap', 'unresolved', 'none')

# sub_type -> (locus, mechanism). Independent of category: a sub-type name
# means the same thing wherever it appears (untested_deferred shows up under
# both Pass and Fail, and describes the same situation in each).
ANNOTATION = {
    # Above the 240 floor, but the dip's split left a group under
    # MIN_GROUP_SPIKES so no L-ratio could be computed on it. locus 'within'
    # because that is where the test would have run; mechanism 'unresolved'
    # because nothing was established about the cluster either way.
    'split_remainder_too_small': ('within', 'unresolved'),
    # Redundant-side. The only sub-type of its category, so the category and the
    # sub-type carry the same information today -- kept as a sub-type anyway
    # because the category is the OUTCOME ("do not count this cluster") while the
    # sub-type is the MECHANISM, and a second mechanism could land in the same
    # outcome later (a partition-detector would).
    're_detection': ('cross-temporal', 'overlap'),
    # Pass-side
    'clean': ('all', 'none'),
    'neighbor_confirmed': ('between', 'contamination'),
    # Fail-side, attributed
    'ldl_bimodal': ('within', 'bimodality'),
    # Rule 1's drift arm. locus 'cross-temporal' like re_detection, and for the
    # same reason: the defect is only visible when spike TIME is considered, not
    # in the feature space alone -- here the two waveform states are early-vs-late
    # rather than interleaved. Mechanism 'drift', distinct from 'overmerge',
    # because the cluster is ONE cell: its own refractory period rules out two.
    'drift': ('cross-temporal', 'time_ordering'),
    'isi_vr': ('temporal', 'refractory'),
    # RENAMED from `noise_contamination` 2026-08-31 (#42, Agrita). The old name
    # asserted a SOURCE -- noise -- that this test never establishes. What the
    # test actually does is compare the unit against a reference channel drawn
    # from its OWN 10-channel group, so it can show that foreign events are
    # present without identifying whose they are. Ground truth says the
    # intruders are in fact other NEURONS (median frac_other 0.081-0.096 vs
    # frac_noise 0.007-0.013), so `noise` was not merely unestablished, it was
    # wrong. The new name states the algorithm's own scope, which is what the
    # locus axis below has said since 2026-07-24.
    'within_cluster_contamination': ('within', 'contamination'),
    'coincident_pair': ('cross-temporal', 'coincidence'),
    'neighbor_boundary': ('between', 'contamination'),
    'neighbor_indistinguishable': ('between', 'contamination'),
    # unattributed / deferred, both sides
    'untested_deferred': ('within', 'unresolved'),
    # RENAMED from `ambiguous_deferred` 2026-08-21 (#33). "Deferred" described a
    # process state -- waiting for MEArec P2 graded contamination -- and that
    # wait is over, so the old name asserted a pending decision that no longer
    # exists. The new name says what the EVIDENCE is: the dip fired and the
    # within-cluster L-ratio did not support it.
    #
    # WHAT THE POPULATION TURNED OUT TO BE. Two ground-truth measurements:
    #   P(this verdict | genuinely CLEAN unit) RISES with n -- 6.7% at n=1000,
    #     7.5% at 2000, 8.2% at 4000 (m260821_dip_null_below_floor.py --large,
    #     400+ clean units). Applied size-by-size to the real 50-session set that
    #     predicts 1,381 such clusters if EVERY cluster were a clean single unit;
    #     1,141 are observed. The population is fully accounted for by clean
    #     units, with none left over to attribute to contamination.
    #   Contamination ANNOUNCES ITSELF AS `overmerged` INSTEAD: against
    #     ground-truth fpos, overmerged runs 54% (fpos 0.10-0.20), 86%
    #     (0.20-0.35), 80% (0.35-0.50), while this verdict is 16.7% then 0% then
    #     0% (m260821_ambiguous_vs_groundtruth.py). Likelihood ratio 1.58, i.e.
    #     weak evidence for contamination.
    # mechanism stays 'unresolved': the dip firing alone does not attribute a
    # mechanism, and 12-15% of clean units at these sizes do it.
    #
    # THE RESIDUAL COST, stated rather than buried: in the fpos 0.10-0.20 band
    # ~17% of genuinely contaminated clusters land here and are not failed on
    # this evidence. Small denominators (43 contaminated clusters, 4 of them
    # here) and synthetic clusters ~3.5x larger than the real population
    # (median 6,961 vs 1,984) both limit that number.
    'dip_only_no_lratio_support': ('within', 'unresolved'),
    'neighbor_conflicting': ('between', 'unresolved'),
    'unexplained': ('open', 'unresolved'),
    # The unit is present in too few drift frames to be judged. locus is
    # TEMPORAL because the evidence is about presence ACROSS TIME (same axis as
    # ISI-VR, which is also about the spike train in time); mechanism is ABSENCE,
    # a new value because none of the existing ones fit -- it is emphatically NOT
    # contamination, and calling it 'unresolved' was what made it look unexplained.
    'intermittent_presence': ('temporal', 'absence'),
    'pending_isi_check': ('temporal', 'unresolved'),
    # Low-spike-count
    'below_threshold': ('n/a', 'none'),
    'ldl_untestable': ('within', 'unresolved'),
}

# Keyed on (tag, decision), NOT bare tag. flagged_ambiguous_ldl and
# flagged_untested_ldl are the two rules that defer to ms_pass and so span
# both decisions -- a bare-tag lookup silently drops one half of each. This
# exact bug was found and fixed in Fig_Population_Stats on 2026-07-31 (the
# panel bars summed to less than their own titled n=); keep the pair key.
_MAP = {
    # --- Pass -------------------------------------------------------------
    ('confirmed_pass', 'PASS'): (PASS, 'clean'),
    ('retrieved_neighbor_confirmed', 'PASS'): (PASS, 'neighbor_confirmed'),
    ('flagged_ambiguous_ldl', 'PASS'): (PASS, 'dip_only_no_lratio_support'),
    ('flagged_untested_ldl', 'PASS'): (PASS, 'untested_deferred'),

    # --- Fail -------------------------------------------------------------
    # confirmed_fail and ldl_bimodal_ms_passed are ONE sub-type: both are
    # rule 1, real bimodal structure. The distinction that matters for the
    # paper (mind_snag already agreed vs mind_snag was overruled by the LDL)
    # is preserved in the tag itself, which classify() passes through.
    #
    # `ldl_bimodal_ms_passed` was `missed_overmerge_corrected` until
    # 2026-08-30. BOTH ARE MAPPED ON PURPOSE: result files already written to
    # data/results carry the old string, and this module raises rather than
    # defaulting on an unknown tag, so dropping it would crash every historical
    # record instead of reading it.
    #
    # THAT OVERCLAIM IS NOW FIXED (2026-09-03). The sub-type is `ldl_bimodal`
    # and the mechanism is `bimodality`: both name the TEST, which found two
    # separable groups, instead of the error it was believed to prove. Ground
    # truth agreed with the old name 19.2% of the time. The error class moved to
    # curate/findings.outcome, where it is measured rather than asserted.
    # What follows is the measurement that made the old name untenable. Measured on MEArec ground truth, the branch names the
    # right mechanism in 6 of its 161 firings -- the structure it detects is
    # real (the dip is not saturated: median p 0.97 on clusters it calls
    # unimodal), but it cannot tell two NEURONS from one neuron in two waveform
    # states. Changing the mechanism moves 161 synthetic clusters out of the
    # Overmerge category, and it entangles with the settled policy of tagging
    # drift units as over-merge, so it is deferred to #53 (hybrid ground truth)
    # to be revisited once rather than twice. See paper_limitations_draft.md
    # §11 and m260828_rule1_override.py.
    ('confirmed_fail', 'FAIL'): (FAIL, 'ldl_bimodal'),
    ('ldl_bimodal_ms_passed', 'FAIL'): (FAIL, 'ldl_bimodal'),
    ('missed_overmerge_corrected', 'FAIL'): (FAIL, 'ldl_bimodal'),  # pre-2026-08-30
    # Rule 1's drift arm (2026-08-30). NOT 'overmerge': this cluster's own spike
    # train shows a single refractory period, so it is ONE cell, and its split is
    # early-vs-late rather than interleaved, so the two waveform states are drift.
    # It still FAILS -- a unit whose waveform walks loses spikes at a
    # time-varying rate, which manufactures apparent rate modulation, and the
    # settled policy is to tag drift rather than correct it. The mechanism is
    # named honestly here BECAUSE it is established, unlike 'overmerge' above.
    ('ldl_bimodal_drift_split', 'FAIL'): (FAIL, 'drift'),
    ('isi_vr_fail', 'FAIL'): (FAIL, 'isi_vr'),
    ('confirmed_fail_within_cluster', 'FAIL'): (FAIL, 'within_cluster_contamination'),
    # Both dip and L-ratio agree the pair overlaps. Split out from the
    # 'pending' bucket below by _neighbor_sub_type().
    ('confirmed_fail_neighbor_real', 'FAIL'): (FAIL, 'neighbor_indistinguishable'),
    ('pending_between_cluster_check', 'FAIL'): (FAIL, None),  # resolved below
    ('unexplained_other_needs_review', 'FAIL'): (FAIL, 'unexplained'),
    ('flagged_ambiguous_ldl', 'FAIL'): (FAIL, 'dip_only_no_lratio_support'),
    ('flagged_untested_ldl', 'FAIL'): (FAIL, 'untested_deferred'),

    # --- Fail, ISI-VR not yet computed ------------------------------------
    # curate()'s '*-pending' branches. Absent from the current 50-session
    # set (passes_isi_independent is populated for all 18974 records), but
    # curate() can still emit them, so they are mapped rather than left to
    # raise on some future partially-computed run.
    # Rule 5's time-domain gate. `declined_coincidence_excess` is a FAIL with a
    # named mechanism: the pair co-fires within 2 ms more than chance, so it is
    # one neuron the sorter split, not two. locus 'cross-temporal' like
    # re_detection and drift -- the defect is only visible when spike TIME is
    # considered, which is the whole reason the gate exists.
    ('declined_coincidence_excess', 'FAIL'): (FAIL, 'coincident_pair'),
    # The deficit branch: FEWER coincidences than chance, which is one neuron's
    # refractory period seen across its two halves. Same sub-type as the excess
    # branch -- both mean "this pair is one unit" -- but a different mechanism,
    # so the TAG differs and the two stay countable apart.
    ('declined_coincidence_deficit', 'FAIL'): (FAIL, 'coincident_pair'),
    ('pending_ccg_check', 'FAIL'): (FAIL, 'pending_isi_check'),
    ('pending_isi_check_within_cluster', 'FAIL'): (FAIL, 'pending_isi_check'),
    ('pending_isi_check_neighbor', 'FAIL'): (FAIL, 'pending_isi_check'),
    ('pending_isi_check_other', 'FAIL'): (FAIL, 'pending_isi_check'),
}

# Tags emitted by curate() BEFORE the 2026-08-31 rename (#42). Kept so that
# results already on disk -- curate_147.json most importantly, 53,428 rows from
# an 18.4-minute run -- still classify without being regenerated. They map to
# the SAME sub-types, so nothing about the taxonomy is duplicated; only the
# spelling of the input differs. Delete once no stored run predates the rename.
LEGACY_TAGS = {
    'confirmed_fail_noise_real': 'confirmed_fail_within_cluster',
    'pending_isi_check_noise': 'pending_isi_check_within_cluster',
}
for _old, _new in LEGACY_TAGS.items():
    for _k, _v in list(_MAP.items()):
        if _k[0] == _new:
            _MAP[(_old, _k[1])] = _v

# The old SUB-TYPE name must NOT be silently accepted: a caller still testing
# `sub_type == 'noise_contamination'` would match nothing and quietly report
# zero rather than fail, which is exactly how a rename becomes a wrong number.
# old name -> (new name, why). The REASON is per entry: one shared sentence
# described the #42 rename and would have been printed, wrongly, for every
# later one.
LEGACY_SUB_TYPES = {
    'noise_contamination': (
        'within_cluster_contamination',
        'renamed 2026-08-31 (#42): the old name asserted a SOURCE the test does '
        'not establish. The within-cluster test never looks at another cluster, '
        'so it cannot tell noise from a neighbour.'),
    'overmerge': (
        'ldl_bimodal',
        'renamed 2026-09-03: the old name asserted an ERROR CLASS, and ground '
        'truth agreed with it 19.2% of the time. What the LDL establishes is '
        'two separable groups; whether they are two NEURONS is a separate claim '
        'it cannot make. The error class now lives in curate/findings.outcome.'),
}

# DELETED, not renamed: the sub-type no longer exists because nothing can
# produce it. Same reasoning as LEGACY_SUB_TYPES -- a caller testing
# `sub_type == 'shape_preserving_decay'` would match nothing and quietly report
# zero, so the name raises instead.
DELETED_SUB_TYPES = {
    'shape_preserving_decay':
        'deleted 2026-09-03: it came from tag `retrieved_noise_gate`, which is '
        'rule 3, RETIRED. Nothing in decision_flow emits that tag and both '
        'corpora carry 0 of them, so the sub-type advertised coverage that did '
        'not exist. Classifying data written BEFORE the retirement will raise '
        'here, which is correct -- those rows predate the current rule table '
        'and must not be scored against it.',
}


def check_sub_type(name):
    """Raise on a retired sub-type name rather than let it match nothing."""
    if name in LEGACY_SUB_TYPES:
        new, why = LEGACY_SUB_TYPES[name]
        raise ValueError(
            f'sub_type {name!r} was renamed to {new!r}: {why} '
            'Update the call site; do not re-add the alias.')
    if name in DELETED_SUB_TYPES:
        raise ValueError(f'sub_type {name!r}: {DELETED_SUB_TYPES[name]}')
    return name

# Display order within each category, most-populated first on the real set.
#
# TWO ENTRIES ARE CURRENTLY EMPTY AND ARE KEPT ON PURPOSE (audited 2026-09-03):
#
#   `unexplained`                 the fall-through for tag
#       `unexplained_other_needs_review`. Empty only because every record now
#       carries the frame-coverage fields that promote it to
#       `intermittent_presence`; a record without them still lands here.
#   `dip_only_no_lratio_support`  from tag `flagged_ambiguous_ldl`. Unreachable
#       only because rule 3 now gives the L-ratio precedence when the dip and
#       the L-ratio conflict, which makes `v == 'ambiguous'` impossible. Revert
#       that precedence and this repopulates.
#
# Both are reachable in principle, so deleting them would remove a real state
# rather than a dead name -- which is the distinction `shape_preserving_decay`
# failed: nothing can emit its tag at all, so it is deleted and registered in
# DELETED_SUB_TYPES.
#
# A sub-type appearing under BOTH Pass and Fail is NOT a duplicate: this dict is
# keyed by CATEGORY and the same tag can yield either decision. See the
# ANNOTATION comment above.
SUB_TYPE_ORDER = {
    PASS: ['clean', 'neighbor_confirmed',
           'untested_deferred', 'dip_only_no_lratio_support'],
    # `drift` was MISSING from this list until 2026-09-03 while being live on
    # both corpora (4 synthetic, 69 real). Consumers index SUB_TYPE_ORDER by
    # category, so an undeclared live sub-type is silently dropped from every
    # figure that iterates it -- the mirror image of the dead labels this pass
    # removed, and the more dangerous of the two: a dead label advertises
    # coverage that does not exist, an undeclared one hides coverage that does.
    FAIL: ['ldl_bimodal', 'isi_vr', 'within_cluster_contamination', 'coincident_pair',
           'drift', 'neighbor_boundary',
           'neighbor_indistinguishable', 'untested_deferred', 'unexplained',
           'intermittent_presence',
           'dip_only_no_lratio_support', 'neighbor_conflicting', 'pending_isi_check'],
    LOW_SPIKE: ['below_threshold', 'split_remainder_too_small', 'ldl_untestable'],
    # One entry today, but the key must exist: every consumer indexes
    # SUB_TYPE_ORDER by category, so a missing key is a KeyError in a dozen
    # figure scripts rather than an empty column.
    REDUNDANT: ['re_detection'],
}

LABELS = {
    'split_remainder_too_small': 'split remainder below the group floor',
    're_detection': 're-detection (already counted elsewhere)',
    'clean': 'clean',
    'neighbor_confirmed': 'neighbour-confirmed',
    'untested_deferred': 'untested (deferred)',
    'dip_only_no_lratio_support': 'dip only, no L-ratio support',
    'ldl_bimodal': 'LDL: two separable groups',
    'isi_vr': 'ISI-VR',
    'within_cluster_contamination': 'within-cluster contamination',
    'coincident_pair': 'coincident pair (one neuron, split)',
    'neighbor_boundary': 'neighbour, boundary',
    'neighbor_indistinguishable': 'neighbour, indistinguishable',
    'neighbor_conflicting': 'neighbour, conflicting signals',
    'unexplained': 'unexplained',
    'intermittent_presence': 'unit absent from most frames',
    'pending_isi_check': 'pending ISI check',
    'below_threshold': f'<{LOW_SPIKE_THRESHOLD} spikes',
    'ldl_untestable': 'LDL untestable',
}


def enrich_drift(records, drift_records):
    """Attach per-cluster drift onto records, mirroring enrich_between().

    Source: data/results/real_drift_vs_overmerge.json, produced by
    m260812_real_drift_vs_overmerge.py from spike_positions.npy. Kept as a merge
    rather than written into the shared decision JSON, which the CatGT thread
    reads.

    Carries the CONTINUOUS drift_ratio, not just the flag: the overmerge risk
    rises smoothly with it (1.6x at 0.5 through 3.2x at 2.0) with no elbow, so a
    boolean alone would throw away most of the information.

    Mutates and returns `records`.
    """
    by_key = {(r['session'], r['clu_id']): r for r in drift_records}
    for r in records:
        src = by_key.get((r['session'], r['clu_id']))
        if src is not None:
            r['drift_ratio'] = src.get('drift_ratio')
            r['drift_um'] = src.get('drift_um')
            r['spread_um'] = src.get('spread_um')
    return records


def enrich_between(records, check_records):
    """Copy dip_pval / l_ratio from the between-cluster check onto records.

    The final-decision records carry only the collapsed verdict
    (between_cluster_confirmed_separated: True/False/None), which cannot
    distinguish the two DIFFERENT situations that both produce None. The raw
    two metrics live in data/results/between_cluster_real_check.json. This
    merges them in rather than mutating that shared file on disk -- the
    CatGT thread reads the same JSONs.

    Mutates and returns `records`.
    """
    by_key = {(r['session'], r['clu_id']): r for r in check_records}
    for r in records:
        src = by_key.get((r['session'], r['clu_id']))
        if src is not None:
            r['dip_pval_between'] = src.get('dip_pval')
            r['l_ratio_between'] = src.get('l_ratio')
            # Needed to redraw the between-cluster projection in figures --
            # without the neighbour's identity the decisive axis cannot be
            # reconstructed, which is why that panel was missing.
            r['neighbor_clu_id'] = src.get('neighbor_clu_id')
            r['n_shared_channels'] = src.get('n_shared_channels')
    return records


def _neighbor_sub_type(r):
    """Resolve curate()'s 'pending_between_cluster_check' bucket.

    It is NOT pending. dip_lratio_between ran on all of these; the verdict
    is None because the two signals landed in opposite corners of the 2x2.
    On the real set the corners are qualitatively different situations and
    the medians are nowhere near each other, so they get separate sub-types:

      dip bimodal + L-ratio high  (n=451, dip p~0, LR 0.135)
          two genuinely distinct units with real contamination at their
          shared boundary. dip runs on the 1-D LDA projection (the single
          best separating direction) while L-ratio measures overlap mass in
          the full ~30-dim PC space, so a pair can show a clean gap on one
          axis and still overlap overall.
      dip unimodal + L-ratio low  (n=73, dip p 0.485, LR 0.003)
          the two tests genuinely conflict; nothing to attribute.

    For comparison the two AGREEING corners are neighbor_confirmed (n=1253,
    dip p~0, LR 0.0001 -> PASS) and neighbor_indistinguishable (n=179, dip p
    0.943, LR 0.373 -> FAIL).

    NOTE deliberately NOT acted on: neighbor_indistinguishable's profile (no
    detectable gap + heavy overlap) is what an OVERSPLIT pair would look
    like. Not promoted to an 'oversplit' mechanism here because dip
    non-significance is a null result, not positive evidence that the two
    are one unit -- that needs its own test.

    Falls back to 'neighbor_conflicting' when the metrics are absent, so a
    caller that skipped enrich_between() still gets a valid (if coarser)
    classification rather than a crash.
    """
    dp, lr = r.get('dip_pval_between'), r.get('l_ratio_between')
    if dp is None or lr is None:
        return 'neighbor_conflicting'
    if dp < DIP_THRESH and lr >= L_THRESH:
        return 'neighbor_boundary'
    if dp >= DIP_THRESH and lr < L_THRESH:
        return 'neighbor_conflicting'
    # Both-agree corners should have been resolved by curate() into
    # retrieved_neighbor_confirmed / confirmed_fail_neighbor_real and never
    # reach this function.
    raise ValueError(
        f'_neighbor_sub_type: dip_pval={dp} l_ratio={lr} is an AGREEING '
        f'corner, so curate() should not have tagged this cluster pending. '
        f'Check that DIP_THRESH/L_THRESH here still match '
        f'm260729_between_cluster_real_check.py.')


def classify(r):
    """Map ONE curate()-tagged record to the 3-category taxonomy.

    Expects the keys curate() returns (tag, decision, needs_inspection) plus
    n_spikes, best_k and is_burst. For the neighbour split, also expects
    dip_pval_between / l_ratio_between -- call enrich_between() first.

    Returns dict:
      category            : 'Pass' | 'Fail' | 'Low-spike-count'
      sub_type            : mutually exclusive within category
      locus               : 'within' | 'between' | 'both' | 'open' | 'n/a'
      mechanism           : 'bimodality' | 'refractory' | 'contamination' |
                            'unresolved' | 'none'
      burst_rescued       : bool  -- overlay, cross-cuts every category
      needs_inspection    : bool  -- overlay, cross-cuts every category
      high_drift          : bool | None -- overlay; None = drift NOT MEASURED,
                            which is not the same as low drift
      drift_ratio         : float | None -- the continuous value, because the
                            risk gradient has no elbow
      underlying_decision : curate()'s raw PASS/FAIL, retained for the
                            Low-spike-count rows where category no longer
                            carries it
      tag                 : passed through, so the finer rule-level
                            distinctions stay reachable downstream
    """
    tag = r['tag']
    decision = r['decision']

    # Low-spike-count takes precedence over EVERY rule (see
    # LOW_SPIKE_THRESHOLD above -- absolute by design, no carve-outs).
    #
    # Two distinct reasons a cluster lands here, kept as separate sub-types
    # because they are not the same claim:
    #   below_threshold -- under LOW_SPIKE_THRESHOLD spikes, so the metrics
    #       are too unstable to judge on. This is the spike-count rule.
    #   ldl_untestable  -- the LDL test could not run at all (best_k is
    #       None) despite having enough spikes to clear the threshold. Not
    #       actually a low-count cluster; the category name is loose for
    #       these, which is why the sub-type names the real reason.
    n_spikes = r.get('n_spikes')
    if n_spikes is not None and n_spikes < LOW_SPIKE_THRESHOLD:
        # STILL FIRST, even ahead of Redundant. A sub-floor cluster that is also
        # a copy is reported as Low-spike-count, because the floor is absolute by
        # design and carving out an exception for one rule reopens exactly the
        # argument that made it absolute. The re-detection evidence is not lost --
        # `redundant_of` stays on the record.
        category, sub_type = LOW_SPIKE, 'below_threshold'
    elif (tag == 'flagged_untested_ldl'
          and 'remainder_too_small' in str(r.get('dip_basis'))):
        # THE 240 FLOOR ASSUMES AN EVEN SPLIT (Agrita, 2026-08-26).
        # LOW_SPIKE_THRESHOLD = 2 * MIN_GROUP_SPIKES is exactly right for a 50/50
        # split, because each group then clears 120. Real splits are lopsided, and
        # the L-ratio needs the SMALL side to clear 120 -- so an 80/20 split needs
        # 600 spikes and a 90/10 split needs 1,200. Below that the dip finds a
        # split and no L-ratio can be computed on it.
        #
        # Measured on the 50 sessions: 31.2% of the 240-400 band was
        # LDL-unadjudicable, 67% of those for `remainder_too_small`, and 89% of
        # rule 8's judged clusters sat under 600 spikes. Those 506 clusters were
        # getting a PASS/FAIL from mind_snag alone with no within-cluster evidence
        # -- the untested-default failure this floor exists to prevent, surviving
        # in the one case the derivation did not cover.
        #
        # THE FIX USES THE RECORDED REASON, NOT A NEW THRESHOLD. `remainder_too_small`
        # already means "a group fell below MIN_GROUP_SPIKES", which IS a
        # spike-count failure, so the spike-count remedy applies by definition.
        # No new constant, and the same 120 that produced the 240.
        #
        # IT NEEDS NO SIZE BOUND, checked rather than assumed: these clusters run
        # median 270 spikes, max 422, 100% under 600. A 99/1 split on a large
        # cluster would be misfiled here, but it never occurs -- a big cluster's
        # split leaves 120 on both sides by construction.
        #
        # 240 REMAINS ABSOLUTE AND UNCHANGED (Agrita's 2026-08-07 call). This is a
        # SECOND, narrower condition beside it, not a replacement. The other 209
        # unadjudicable clusters (full/large_unadjudicable, median 468, max 5,409)
        # stay judged -- their failure is not about spike count and must not
        # borrow the spike-count remedy.
        #
        # GATED ON rule 8's TAG, WHICH THE FIRST VERSION WAS NOT, and that was a
        # real over-reach: 56 of 495 clusters caught by the bare dip_basis test had
        # a POSITIVE finding from a test that did run -- 51 failed ISI-VR and 5 are
        # re-detection copies. `remainder_too_small` invalidates the LDL and nothing
        # else; ISI-VR counts intervals and the re-detection test counts
        # coincidences, neither of which needs 120 spikes per group. Filing those as
        # "low spike count" would discard evidence we have.
        #
        # This is deliberately ASYMMETRIC with the 240 floor, which DOES override
        # ISI-VR. Below 240 a spike train yields too few intervals for ISI-VR to be
        # reliable either; at 240-422 it is fine. So the absolute floor overrides
        # everything and this one overrides only the rule that rests on the LDL.
        category, sub_type = LOW_SPIKE, 'split_remainder_too_small'
    elif tag == 'redundant_re_detection':
        category, sub_type = REDUNDANT, 're_detection'
    elif tag == 'flagged_untested_ldl' and r.get('best_k') is None:
        category, sub_type = LOW_SPIKE, 'ldl_untestable'
    else:
        try:
            category, sub_type = _MAP[(tag, decision)]
        except KeyError:
            raise KeyError(
                f'taxonomy.classify: unmapped (tag, decision) pair '
                f'({tag!r}, {decision!r}). curate() gained a branch this '
                f'module does not know about -- add it to _MAP rather than '
                f'letting it fall through to a default bucket.') from None
        if sub_type is None:  # pending_between_cluster_check
            sub_type = _neighbor_sub_type(r)
        # `unexplained` was never a mystery -- it was a MEASUREMENT WE HAD NOT
        # MADE. These clusters fail because the unit is ABSENT from most drift
        # frames: mind_snag counts a frame with no spikes against the clean
        # fraction ("unit had no spikes = drifted away = not isolated"), so a
        # cluster absent from >40% cannot reach 0.6 however clean its present
        # frames are, while classify_dominant_source SKIPS absent frames and so
        # finds no source. Both correct; they measure different things.
        #
        # EVIDENCE-BASED, not a blanket rename: it re-labels only clusters whose
        # own numbers show absence AND cleanliness where present. A future
        # cluster with the same tag but genuine unattributed contamination keeps
        # `unexplained`, which is what that bucket should mean.
        # Measured on all 44: absent 43-95% of frames, max L-ratio over present
        # frames 0.007 median / 0.186 max (threshold 0.2), and 44/44 would PASS
        # if judged only where the unit is present.
        if sub_type == 'unexplained':
            absent = r.get('absent_frame_frac')
            clean_present = r.get('clean_frac_when_present')
            if (absent is not None and clean_present is not None
                    and absent > ABSENT_FRAME_FRAC_THRESH
                    and clean_present >= ISOLATED_T_RATIO):
                sub_type = 'intermittent_presence'

    locus, mechanism = ANNOTATION[sub_type]
    return {
        'category': category,
        'sub_type': sub_type,
        'locus': locus,
        'mechanism': mechanism,
        'burst_rescued': bool(r.get('is_burst')),
        # SINGLE IMPLEMENTATION IN curate(): clusters under LOW_SPIKE_THRESHOLD
        # cannot be flagged for review, because no test can ever resolve them
        # (see decision_flow.curate for the measured reason). Applied there, not
        # here, so `r['needs_inspection']` and this row can never disagree.
        'needs_inspection': bool(r.get('needs_inspection')),
        # Third overlay, added 2026-08-13. Cross-cuts every category like the
        # other two: drift raises the false-positive rate of the overmerge test
        # (drift-free 6.9% -> 33-36% on ground truth, fpos-corrected 2026-08-20;
        #  the drift arms barely moved under the correction, so the drift
        #  effect is real and not an artefact) and we CANNOT correct it -- see
        # curate.constants.DRIFT_RATIO_FLAG. None means drift was not measured
        # for this cluster, which is NOT the same as low drift.
        'high_drift': (None if r.get('drift_ratio') is None
                       else bool(r['drift_ratio'] >= DRIFT_RATIO_FLAG)),
        'drift_ratio': r.get('drift_ratio'),
        'underlying_decision': decision,
        'tag': tag,
    }


def curate_and_classify(records):
    """Run curate(), then taxonomy, on every record. THE ONLY WAY TO RUN curate().

    Agrita, 2026-09-02: "we should make a rule to always run taxonomy after
    curate."

    WHY IT IS A RULE AND NOT A CONVENTION. curate() decides; taxonomy DESCRIBES.
    A run that stores only curate()'s output looks complete -- every cluster has
    a decision, a rule and a tag -- while the description layer has never been
    asked whether it can name what happened. Nothing errors, so the gap is
    invisible.

    That is exactly how A4 hid. `intermittent_presence` had existed in this file
    since 2026-08-21, correctly gated on `absent_frame_frac` and
    `clean_frac_when_present`, and it fired on 41 real clusters. On the synthetic
    corpus it fired on ZERO -- not because the corpus lacks intermittent units
    (it has a recording built of them) but because the synthetic pipeline never
    computed those two inputs, and no run ever invoked taxonomy to notice. Seven
    genuinely clean units sat in `unexplained` -- "no rule understood this" --
    for eleven days, in the corpus's weakest family, while the decision layer
    was working perfectly.

    Running both together means a missing taxonomy INPUT shows up as a
    `unexplained` count that will not go down, in the same place the decision
    counts are read, instead of nowhere at all.

    ALL THREE LAYERS TRAVEL TOGETHER, and that is not tidiness. Findings read
    enrichments (`ccg_accusers` above all), so computing them at a different
    moment from curate() means computing them on a different record. Doing
    exactly that is how the first version of this merge produced 1,129
    `clean_untested_channels` and zero `clean`: curate_session evaluated
    findings before the run script attached `ccg_accusers`, so `coincident` read
    as untested for every cluster in the corpus. Re-running this function after
    an enrichment lands updates all three, or none.

    Returns the records, updated in place with all three layers' fields.
    """
    from unitassay.curate.decision_flow import curate
    from unitassay.curate.findings import findings, outcome
    for r in records:
        r.update(curate(r))
    for r, c in zip(records, classify_all([dict(x) for x in records])):
        r.update(category=c['category'], sub_type=c['sub_type'],
                 locus=c['locus'], mechanism=c['mechanism'])
    for r in records:
        f = findings(r)
        r['findings'] = sorted(f['fired'])
        r['untested_channels'] = sorted(f['untested'])
        r['outcome'] = outcome(f)
    return records


def classify_all(records):
    """classify() over an iterable, returning a list of merged dicts."""
    return [dict(r, **classify(r)) for r in records]


# ── the one place allowed to say whether L-ratio failed ──────────────────
#
# THIS FUNCTION EXISTS BECAUSE THE SAME BUG KEEPS COMING BACK. Figure after
# figure has re-derived "did L-ratio fail?" inline, and each time it has
# ended up asserting an L-ratio failure for clusters where mind_snag's
# L-ratio bars are entirely BELOW threshold -- clusters mind_snag failed on
# ISI-VR instead. Agrita has caught this more than once. Do not inline this
# logic again; call lratio_state().
#
# The trap: `dominant_source` is NOT "which L-ratio comparison failed" for
# every cluster. m260723_catgt_devcopy_stable_fail_audit.py:351 computes it
# ONLY for `(not ms_pass) and ldl_verdict in CLEAN_VERDICTS`. For every other
# record it is absent -- 4,107 judged clusters as of 2026-08-07 (2,646
# overmerge, 761 isi_vr, 384 untested, 223 unexplained, 93 ambiguous).
# Absent does NOT mean "L-ratio failed and we don't know the source"; it
# means NOT EVALUATED. Treating None/'unclassified' as a failure is the bug.
#
# classify_dominant_source() can also return 'none' (nothing crossed
# threshold -> L-ratio genuinely PASSED even though ms_pass is False) and
# 'mixed' (noise and neighbour both crossed, neither dominates). Both need
# their own branch; neither is a noise or neighbour failure.
LRATIO_STATES = ('pass', 'noise_fail', 'neighbor_fail', 'mixed_fail', 'not_evaluated')


def n_affirmative_channels(r):
    """How many of the three channels POSITIVELY cleared this cluster?

    A channel is affirmative only if it ran and passed. A failure that was later
    EXPLAINED AWAY does not count:

      LDL      resolved AND ldl_verdict == 'not_overmerged'
               (an untestable channel is not a pass)
      L-ratio  lratio_passed_measured is True
               (a retrieval says "the failure is explained", not "it passed")
      ISI-VR   passes_isi_independent AND NOT is_burst
               (the burst rescue forgives violations; it does not clear them)

    Returns None if any channel's evidence is missing, so absence is never
    counted as a pass -- the same rule as is_resolved() and lratio_state().
    """
    if r.get('lratio_passed_measured') is None or r.get('dip_basis') is None:
        return None
    from unitassay.curate.bimodality import is_resolved
    ldl = bool(is_resolved(r.get('dip_basis'))
               and r.get('ldl_verdict') == 'not_overmerged')
    lr = bool(r.get('lratio_passed_measured'))
    isi = bool(r.get('passes_isi_independent') is True and not r.get('is_burst'))
    return int(ldl) + int(lr) + int(isi)


def check_pass_has_affirmative_channel(rows):
    """INVARIANT: every PASS has >=1 channel that affirmatively cleared it.

    Not a filter -- an invariant the pipeline ALREADY satisfies (0 violations of
    8,463 passes, measured 2026-08-20). It exists so a future rule change cannot
    quietly introduce "pass by exhaustion of objections": a cluster passing
    because the LDL test could not run, the L-ratio failure was retrieved, and
    the ISI-VR violations were burst-forgiven -- three explanations and no
    evidence.

    That is not hypothetical. The #17 fall-through dry run would have introduced
    exactly 13 such passes, all small clusters (240-864 spikes, 12 of 13 under
    400) carrying violation rates up to 105/s. Requiring one affirmative channel
    blocks precisely those 13 and keeps the other 145 rescues.

    Returns the offending records, so a caller can assert on emptiness.
    """
    bad = []
    for r in rows:
        if r.get('category') != PASS:
            continue
        n = n_affirmative_channels(r)
        if n == 0:
            bad.append(r)
    return bad


def lratio_state(r):
    """Return one of LRATIO_STATES for a record. Never guesses.

    'pass'          mind_snag's combined criterion passed, so L-ratio passed
                    (ms_pass = passes_lratio AND passes_isi).
    'noise_fail'    noise-channel comparison dominates the failure.
    'neighbor_fail' a different cluster dominates the failure.
    'mixed_fail'    both crossed threshold, neither dominates.
    'not_evaluated' dominant_source was never computed for this cluster, or
                    the audit could not load its SortData. Say NOTHING about
                    L-ratio for these -- mind_snag may well have failed them
                    on ISI-VR alone.
    """
    if r['ms_pass']:
        return 'pass'
    dom = r.get('dominant_source')
    if dom == 'noise':
        return 'noise_fail'
    if dom == 'other_cluster':
        return 'neighbor_fail'
    if dom == 'mixed':
        return 'mixed_fail'
    if dom in ('none', 'lratio_passed'):
        # Nothing crossed threshold: L-ratio PASSED, so mind_snag failed this
        # cluster on the OTHER criterion (ISI-VR).
        #
        # 'none' was the value classify_dominant_source() returns; 'lratio_passed'
        # is what the audit now STORES for it. Before 2026-08-12 the audit
        # collapsed 'none' into 'unclassified', so this branch was dead code and
        # these clusters were reported as not_evaluated — i.e. "we don't know"
        # instead of "L-ratio explicitly passed every frame". 967+ clusters were
        # affected. See m260812_recompute_dominant_source.py.
        return 'pass'
    # 'unclassified' (legacy) and 'not_computable' both mean the attribution
    # could not be computed. Say nothing about L-ratio for these.
    return 'not_evaluated'


def check_lratio_claims(records):
    """Guard against the recurring mislabel. Returns a list of violations.

    Cross-checks lratio_state() against `frac_fail` -- the fraction of drift
    frames whose L-ratio crossed threshold, i.e. literally the bars drawn in
    column 6 of the diagnostic figures. Where frac_fail is available the two
    must agree, so a figure caption can never contradict the bars beside it.
    Call this from any script that labels L-ratio status.
    """
    bad = []
    for r in records:
        ff = r.get('frac_fail')
        if ff is None:
            continue
        st = lratio_state(r)
        if st in ('noise_fail', 'neighbor_fail', 'mixed_fail') and ff == 0:
            bad.append((r.get('session'), r.get('clu_id'), st, ff,
                        'claims L-ratio failure but no frame crossed threshold'))
        elif st == 'pass' and ff > 0:
            bad.append((r.get('session'), r.get('clu_id'), st, ff,
                        'claims L-ratio pass but frames crossed threshold'))
    return bad
