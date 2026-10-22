#!/usr/bin/env python3
"""curate/decision_flow.py

curate(): executable implementation of the 5-stage decision-flow rule table
finalized 2026-07-23/24. The rule table had existed only as a written spec;
this module is its one implementation.

Input contract: curate() takes ONE merged record (a dict) per cluster, with
these keys expected (missing/None keys are handled explicitly, not assumed):
  ldl_verdict   : 'unimodal' | 'not_overmerged' | 'overmerged' | 'multiway' |
                  'ambiguous' | 'untested'
  screen_pass       : bool -- the screen layer's real curation decision
  dominant_source : 'noise' | 'other_cluster' | 'unclassified' | None --
                  the screen layer's OWN vocabulary, left unchanged so this field
                  stays in step with it. Read it for what the algorithm
                  compared, not for what contaminates: 'noise' means the
                  within-cluster reference-channel comparison dominated, i.e. no
                  candidate SOURCE was named -- NOT that the contaminant is
                  noise. Ground truth says it is usually another neuron.
                  Our own tags say this properly; see curate/taxonomy.py.
                  which L-ratio comparison dominates the failure, only
                  meaningful when screen_pass=False and ldl_verdict is clean
  noise_ch_flagged : bool | None -- clus_channel_info()'s flag (no
                  shape-dissimilar reference channel existed at all)
  neighbor_never_fails : bool | None -- STRICT: neighbor L-ratio never
                  crossed threshold in ANY frame
  passes_isi_independent : bool | None -- the screen layer's real ISI-VR/burst-
                  corrected criterion, recomputed independently. None means
                  not yet computed for this record (conservative default
                  applies, tagged as pending, not silently assumed).
  isi_rate      : float | None -- RAW fraction of intervals under 1 ms
                  (n_viol / (n_spikes-1)). Deliberately NOT the Hill ratio and
                  NOT thresholded: `passes_isi_independent` is the yes/no that
                  the screen derives from this and then throws away, and the
                  continuous value still discriminates among clusters that all
                  passed that flag. Rule 1 only. None = never computed, which
                  cannot be read as "one cell", so the over-merge stands.
  split_time_auc : float | None -- how time-ordered the LDL's own split is, as
                  an AUC of spike time between the two groups on the LDA axis
                  the dip ran on. 0.5 = interleaved, 1.0 = early-vs-late.
                  Distinguishes a drifting unit from a burst/waveform-state
                  split. Rule 1 only. None cannot assert "not drift", so the
                  cluster is held in the drift arm rather than released.
  between_cluster_confirmed_separated : bool | None -- real dip_lratio_between()
                  result on this cluster vs its actual dominant neighbor
                  cluster (channel-aligned, min_group_spikes=120 guard),
                  computed for the unexplained_fail_neighbor / passes-ISI-VR
                  population. True =
                  both dip and L-ratio confirm real separation (retrieve).
                  False = both confirm genuine overlap (fail stands). None =
                  signals disagree, or not yet computed for this record.

Returns dict: {decision: 'PASS'|'FAIL', needs_inspection: bool, tag: str, rule: str}
tag identifies WHY, for downstream panel breakdowns (e.g. distinguishing
'confirmed_fail' from 'missed_overmerge_corrected', both of which resolve to
the same final FAIL decision but come from different original categories).

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""

from unitassay.curate.constants import (LOW_SPIKE_THRESHOLD, ccg_z_threshold,
                             ccg_deficit_alpha, poisson_cdf)

_MISSING = object()      # distinguishes "absent" from "present and None"
CLEAN_VERDICTS = ('unimodal', 'not_overmerged')
OVERMERGE_VERDICTS = ('overmerged', 'multiway')

# Rule 1's three-way (see the long note at the branch). Both thresholds are
# DERIVED, not chosen, and both are stated here rather than inline so a
# re-derivation on new data has one place to land.
#
# ISI_ONE_CELL_FLOOR -- raw fraction of intervals under 1 ms below which the
#   spike train is consistent with a single cell. Swept 0.0001-0.002 on the
#   synthetic corpus; significant across the whole plateau, and 0.0002 is the
#   most conservative point, losing zero over-merges corpus-wide.
# SPLIT_TIME_ORDERED_AUC -- above this the split is early-vs-late, i.e. drift.
#   The ~98th percentile of MEArec's own firings, which serve as a drift-free
#   null because MEArec's drift is too weak to time-order these splits at all.
#
# NEITHER TRANSFERS AUTOMATICALLY. The far-pair/Poisson-null work established
# that a calibration fitted on one corpus does not carry to another; these are
# calibrated on MEArec and checked on 50 real sessions, and a new probe or
# preparation needs the derivation re-run, not these numbers copied.
ISI_ONE_CELL_FLOOR = 0.0002
SPLIT_TIME_ORDERED_AUC = 0.6


# THE RULES, in the order a reader would check them -- which is NOT the order
# the legacy ids were assigned in.
#
# These ARE the rules and figures name them that way; the field is called `stage` only because `rule` is
# taken by the legacy id, which is frozen (14 analysis scripts select on it with
# startswith('1') for over-merge and startswith('5') for neighbour, so
# renumbering under them would silently change what they select). The ids are archaeology -- they number the
# order we FOUND the problems in, so rule 0 fires third, rule 2 (the only PASS)
# ninth, rule 3 survives only as `3-pending`, and there is no rule 6.
#
# `stage` is added ALONGSIDE `rule`, not instead of it. 14 analysis scripts
# select on the id with `str(r['rule']).startswith('1')` (over-merge) or
# `startswith('5')` (neighbour). Renumbering so stage 1 is IDENTITY would make
# those silently select duplicates instead -- no error, wrong numbers. `rule` is
# therefore frozen as a legacy key and every new figure keys on `stage`.
#
# A STAGE IS A SCOPE, NOT AN OUTCOME. It says what the test
# could SEE, not what the cluster IS: the within-cluster test is oblivious to
# other clusters, so what it finds is "not this unit" and nothing more, while
# the between-cluster test is the only one that can say "other cluster". Several
# stages reach the same outcome -- contamination, refractory and composition are
# all majority over-split -- and the bridge between the two vocabularies is
# MEASURED purity, never a definition. See curate/findings.py.
STAGE = {
    'R': '1-identity',
    '0': '2-refractory',
    '1': '3-composition', '1-drift': '3-composition',
    '4': '4-contamination', '7-within': '4-contamination',
    '3-pending': '4-contamination',
    '5': '5-neighbour', '5-all': '5-neighbour', '5-pending': '5-neighbour',
    '5-all-pending': '5-neighbour', '5-pending-isi': '5-neighbour',
    '5-ccg': '5-neighbour', '5-ccg-deficit': '5-neighbour',
    '5-ccg-pending': '5-neighbour',
    '7-neighbor': '5-neighbour', '7-neighbor-retrieved': '5-neighbour',
    '7-neighbor-all': '5-neighbour',
    '7': '6-unresolved', '8': '6-unresolved',
    '2': '7-clean',
    'other': '8-fall-through', 'other-pending': '8-fall-through',
}

# ONE NAMING SCHEME: rule names follow one logical order, not a mix of
# numerical and alphabetical ids.
#
# There were THREE schemes running at once -- the legacy id (`R`, `0`, `1-drift`,
# `5-ccg`, `7-within`, `other`), the `stage` field, and the figure titles -- and
# the legacy SUFFIXES mixed three unrelated ideas: mechanism (`-drift`, `-ccg`,
# `-within`), scope (`-all`) and state (`-pending`).
#
# THE SCHEME: `<rule number><letter>` where the number is POSITION IN THE FLOW
# and the letter is WHICH TEST within that rule, lettered in firing order. The
# label after it always names WHAT WAS TESTED, never the state it left the
# cluster in. A rule with one test has no letter.
#
# Several legacy ids collapse onto one canonical sub-rule, which is correct:
# `5`, `5-all`, `7-neighbor` and `7-neighbor-retrieved` all run the SAME
# feature-space test and differ only in how the cluster reached it. The legacy
# id stays as the internal key -- 14 scripts select on it with startswith() --
# but nothing shown to a reader should use it.
# LABELS (the three-decision rule leaves no "pending"/"untested" state, so no
# label may name one):
#   4b  'evidence incomplete' -> 'reference not cleared'
#   5d  'evidence incomplete' -> 'neighbour not cleared'
#   8   'unattributed' / FALL-THROUGH -> 'stability below 60%' / STABILITY
#       -- verified on the 147 (69d20fa): all 726 rule-8 clusters have isolation
#       stability < 0.6 (median 0.14) with no L-ratio window over threshold and
#       the independent ISI check passed, so stability IS the failing criterion.
#   6   ('no verdict', UNRESOLVED) fires 0 times and is NOT DRAWN in figures;
#       FIGURE_RULES below is the set a figure may show.
# Only labels changed; the legacy keys are frozen (see above).
RULE_ID = {
    'R': ('1', 'overlap and lag'),
    '0': ('2', 'refractory ratio'),
    '1': ('3a', 'bimodality'),
    '1-drift': ('3b', 'time-ordered split'),
    '4': ('4a', 'reference channel'),
    '7-within': ('4a', 'reference channel'),
    '3-pending': ('4b', 'reference not cleared'),
    '5': ('5a', 'feature space'),
    '5-all': ('5a', 'feature space'),
    '7-neighbor': ('5a', 'feature space'),
    '7-neighbor-retrieved': ('5a', 'feature space'),
    '7-neighbor-all': ('5a', 'feature space'),
    '5-ccg': ('5b', 'coincidence excess'),
    '5-ccg-deficit': ('5c', 'refractory deficit'),
    '5-pending': ('5d', 'neighbour not cleared'),
    '5-all-pending': ('5d', 'neighbour not cleared'),
    '5-ccg-pending': ('5d', 'neighbour not cleared'),
    '5-pending-isi': ('5d', 'neighbour not cleared'),
    '7': ('6', 'no verdict'),
    '8': ('6', 'no verdict'),
    '2': ('7', 'all rules passed'),
    'other': ('8', 'stability below 60%'),
    'other-pending': ('8', 'stability below 60%'),
}

RULE_TITLE = {
    '1': 'IDENTITY', '2': 'REFRACTORY', '3': 'COMPOSITION',
    '4': 'CONTAMINATION', '5': 'NEIGHBOUR', '6': 'UNRESOLVED',
    '7': 'CLEAN', '8': 'STABILITY',
}

# the rules a figure draws: rule 6 never fires and is omitted
FIGURE_RULES = ('1', '2', '3', '4', '5', '7', '8')


def rule_id(rule):
    """Canonical id for a legacy rule id, e.g. '5-ccg-deficit' -> '5c'."""
    try:
        return RULE_ID[str(rule)][0]
    except KeyError:
        raise KeyError(
            f'rule {rule!r} has no canonical id. A new rule must be placed in '
            f'RULE_ID deliberately -- defaulting it would print an id that '
            f'sorts into whatever position the string happens to take.') from None


def rule_label(rule, with_title=False):
    """Display label, e.g. '5c refractory deficit' (or 'RULE 5c NEIGHBOUR ...')."""
    rid, what = RULE_ID[str(rule)]
    n = rid[0]
    return (f'RULE {rid} {RULE_TITLE[n]} — {what}' if with_title
            else f'{rid} {what}')


STAGE_QUESTION = {
    '1-identity':      'already counted under another cluster?',
    '2-refractory':    'can one neuron produce these spikes?',
    '3-composition':   'two populations inside it?',
    '4-contamination': 'polluted by something this cluster alone can see?',
    '5-neighbour':     'separable from the neighbours it is tested against?',
    '6-unresolved':    'could it be tested at all?',
    '7-clean':         'nothing found',
    '8-fall-through':  'nothing claimed it',
}


def stage_of(rule):
    """Stage for a rule id. Raises rather than inventing one for a new rule."""
    try:
        return STAGE[str(rule)]
    except KeyError:
        raise KeyError(
            f'rule {rule!r} has no stage. A new rule must be placed in STAGE '
            f'deliberately -- defaulting it would put it in whatever stage the '
            f'figure happened to draw last.') from None


def curate(r):
    """Apply the rule table, then the one override that outranks all of it.

    Split into a wrapper plus `_apply_rules` so the low-spike override has EXACTLY
    ONE implementation. It was first written in taxonomy.classify(), which left
    `r['needs_inspection']` at True while the taxonomy row said False -- two
    answers to one question, and every consumer reading the record directly got
    the wrong one.
    """
    out = _apply_rules(r)
    # A CLUSTER UNDER THE FLOOR CANNOT BE FLAGGED FOR REVIEW (2026-08-21).
    # Several rules set needs_inspection=True meaning "a test still owes us an
    # answer". Below LOW_SPIKE_THRESHOLD no test can ever give one: the LDL
    # L-ratio needs ~120 spikes per group in 30-dim, and the dip cannot stand in
    # for it -- ground-truth type-1 21% at n=120, 54% at n=60, with 2.4-5.1% of
    # KNOWN CLEAN units returning dip p ~= 0 at every n, so no positive
    # threshold reaches 5%. The conjunction does not help either: below the floor
    # the L-ratio fires on 92-100% of clean units, so it adds no specificity and
    # the conjunction collapses onto the dip. Measured on 414 ground-truth clean
    # units across 16 drift-free recordings,
    # a null that reproduces the adopted operating point (0.043 vs Phase 5's
    # 0.047-0.059) where it can be checked.
    #
    # So "not judged" is a RESULT here, not a backlog item, and asking a human to
    # resolve 1,955 clusters that no statistic can decide is asking for work with
    # no possible outcome. The DECISION is untouched -- only the flag.
    if out['needs_inspection']:
        n_spikes = r.get('n_spikes')
        if n_spikes is not None and n_spikes < LOW_SPIKE_THRESHOLD:
            out['needs_inspection'] = False
    # `stage` IS NOT WRITTEN TO THE ROW (2026-09-08). It was fully derivable
    # from `rule_id` -- the canonical number IS the stage number, 0 disagreements
    # across 53,428 clusters, and the test suite asserts it -- so carrying both
    # was a second copy of one fact that could only ever go out of sync. Callers
    # that want it derive it on read: `stage_of(row['rule'])`, or
    # `STAGE_QUESTION[stage_of(row['rule'])]` for the question that stage asks.
    # Over-information is also harmful.

    # THE READER-FACING ID, emitted alongside the internal one.
    #
    # `rule` is the LEGACY key and it stays, because 14 scripts select on it
    # with startswith(). But it is not a name anybody should read: its numbers
    # COLLIDE with the documented scheme -- legacy '2' is a clean cluster while
    # documented rule 2 is REFRACTORY, legacy '1' is bimodality while documented
    # rule 1 is IDENTITY, and identity and refractory are emitted as 'R' and
    # '0'. A reader who trusted the rule table and then opened a row misread
    # every one of them.
    #
    # So the canonical id and its label are written to the row too. They were
    # computed all along and thrown away at the exit; nothing else changes, and
    # `rule_id` is what the README's table is indexed by.
    out['rule_id'] = rule_id(out['rule'])
    out['rule_label'] = rule_label(out['rule'])
    return out


def _apply_rules(r):
    v = r['ldl_verdict']
    # PRECEDENCE WHEN THE TWO STATISTICS CONFLICT: the L-ratio wins.
    # A conflict between the L-ratio and the LDL needs an answer: these are not
    # low-spike clusters, which are already filtered out.
    #
    # `ambiguous` is `separation_verdict`'s name for the dip test and the
    # L-ratio contradicting each other, and on the 273 such clusters the
    # disagreement is one-sided: the dip calls SEPARATED on 272 of them and the
    # L-ratio calls NOT separated. Against ground truth the L-ratio is right
    # 256/273 and the dip 17/273 -- the 256 reproducing the Phase 5 number
    # exactly. Median 1,014 spikes, minimum 260, so the floor is not the issue.
    #
    # STATED AS PRECEDENCE, NOT AS POWER. Only 16 of the 273 are truly
    # over-merged, so "never call over-merge here" is right 94.1% by saying
    # nothing, and the L-ratio's 93.8% IS that base rate -- it takes the majority
    # side on 272 of 273 (score against the conditional base rate). What the numbers
    # establish is that the DIP is unreliable in this regime, not that the
    # L-ratio has resolved the question. Which 16 are the real over-merges stays
    # unknown, consistent with the within-cluster CCG (AUC 0.476, not powered)
    # and the auto-correlogram (AUC 0.547, powered but not specific).
    #
    # COSTS ALMOST NO DECISION: 271 of 273 keep the same verdict, because rule 7
    # was deferring to the screen and the screen already agreed with the rule that
    # now names them. The 2 that move are both PASS -> FAIL and both correct.
    # Balanced accuracy 83.7 -> 83.7. This is an ATTRIBUTION fix: the clusters
    # stop reading "ambiguous, deferred" and get named by the rule that decides
    # them -- 84 duplicate, 61 contamination, 27 refractory, 80 confirmed pass.
    if v == 'ambiguous':
        v = 'not_overmerged'
    screen_pass = bool(r['screen_pass'])
    passes_isi_ind = r.get('passes_isi_independent')  # None = not yet checked

    # Rule 1: LDL overmerged/multiway -> FAIL always, overriding the screen,
    # UNCONDITIONALLY -- checked before the ISI-VR gate below on purpose
    # (2026-07-31). This project's central finding is that the LDL/L-ratio
    # overmerge test catches real curation errors the screen layer alone
    # (which already includes ISI-VR) misses -- that's the reason LDL is
    # combined with the screen at all, not a competing method. If a
    # cluster showing real overmerge structure ALSO happens to fail ISI-VR,
    # letting ISI-VR claim the tag would understate what the overmerge test
    # itself is catching. So overmerge attribution is preserved in full
    # here; only the categories with no such competing finding to protect
    # (ambiguous/untested, rules 7/8 below) defer to the ISI-VR gate.
    # Collapses old confirmed_fail (screen_pass already False, agrees) and the
    # screen_pass-overriding branch (screen_pass was True) into one final FAIL.
    #
    # THE OVERRIDE BRANCH IS NAMED AND FLAGGED.
    # It used to be `missed_overmerge_corrected`, which asserts a mechanism the
    # test does not establish. Measured against MEArec ground truth:
    # it fires 161 times, catches 6 true
    # over-merges and destroys 66 ground-truth-clean single units -- 41% of its
    # firings, and 11 clean units per over-merge recovered. Conditioned on the
    # 18 over-merges the screen actually passed its sensitivity is a respectable
    # 33.3%, so the branch is not broken; the NAME was.
    #
    # What the LDL establishes is that the spikes form two statistically
    # separable groups. That is true for all 161. "Over-merged" adds a second
    # claim -- that the two groups are two NEURONS -- and examples show why it
    # fails: the casualties
    # are genuinely bimodal, but they are ONE neuron in two waveform states
    # (burst shape change, drift). Ground truth says one planted neuron owns a
    # median 99.3% of their spikes. In PC space the two cases are the same
    # picture, which is why nothing in the feature space separates them
    # (dip_pval AUC 0.535, l_ratio 0.521, lopsidedness 0.570, split
    # time-ordering 0.571 -- all chance).
    #
    # So the decision stays FAIL -- we cannot currently tell the two apart, and
    # shipping an over-merge is the worse error -- but it is now marked
    # needs_inspection, because FAIL alone silently discards a real neuron 41%
    # of the time. The flag costs no sensitivity (all 6 catches are still
    # caught) and makes the tool's uncertainty visible instead of resolving it
    # by fiat.
    #
    # NOT changed here, deliberately: curate/taxonomy.py still maps this tag to
    # mechanism 'overmerge'. Same overclaim one layer down, but it interacts
    # with the settled policy of tagging drift units as over-merge, and 6
    # positives cannot settle it. Deferred to hybrid ground truth so the
    # category is revisited once, not twice.
    if v in OVERMERGE_VERDICTS:
        if not screen_pass:
            return {'decision': 'FAIL', 'needs_inspection': False,
                    'tag': 'confirmed_fail', 'rule': '1'}

        # --- THE OVERRIDE IS NOW A THREE-WAY, NOT A VERDICT (2026-08-30) ------
        # The LDL establishes that the spikes form two separable groups. It does
        # NOT establish that they are two NEURONS -- measured on MEArec ground
        # truth, this branch fired 161 times, caught 6 true over-merges and
        # rejected 66 clean single units. No PC-space statistic separates those
        # two outcomes (dip AUC 0.535, l_ratio 0.521, lopsidedness 0.570), and
        # the dip is not at fault: it is not saturated (median p 0.97 on the
        # clusters it calls unimodal), so the bimodality is real and the failure
        # is attribution. Two quantities OUTSIDE the feature space resolve it,
        # and both are already computed upstream -- no extra pass, no new data.
        #
        # 1. TIMING SAYS HOW MANY CELLS. One neuron cannot fire twice inside its
        #    refractory period; two neurons sharing a cluster do not coordinate,
        #    so some of their spikes land close together. This is the RAW rate,
        #    which the screen's own thresholded, burst-corrected flag computes and
        #    then discards -- and it discriminates (AUC 0.846) even though every
        #    cluster reaching here already PASSED that flag. On synthetic:
        #    29 correct decision flips against 9 wrong, McNemar p=0.0017, and at
        #    this floor zero over-merges are lost corpus-wide (69/77 unchanged).
        #
        # 2. TIME-ORDERING SAYS WHETHER "ONE CELL" MEANS DRIFT. A drifting unit
        #    splits early-vs-late; a burst/waveform-state split is interleaved.
        #    Scoping on drift_ratio instead was REJECTED: it is a correlate
        #    (rho +0.425 with time-ordering) and the two disagree on 68 of 215
        #    real clusters, in both directions. SPLIT_TIME_ORDERED_AUC is
        #    calibrated, not chosen: MEArec's drift is too weak to time-order
        #    these splits at all (98.1% of its firings fall below 0.6), so it
        #    serves as a drift-free null and 0.6 is its ~98th percentile.
        #
        # ABSENT IS NOT NEGATIVE. If isi_rate was never computed we cannot claim
        # "one cell", so the override stands -- the recurring error class in this
        # codebase (dominant_source, high_drift, rule 5's None) is not repeated
        # here. Likewise a missing split_time_auc cannot assert "not drift", so
        # it holds the cluster in the drift arm rather than releasing it.
        isi_rate = r.get('isi_rate')
        if isi_rate is None or isi_rate >= ISI_ONE_CELL_FLOOR:
            return {'decision': 'FAIL', 'needs_inspection': True,
                    'tag': 'ldl_bimodal_ms_passed', 'rule': '1'}
        t_auc = r.get('split_time_auc')
        if t_auc is None or t_auc > SPLIT_TIME_ORDERED_AUC:
            return {'decision': 'FAIL', 'needs_inspection': True,
                    'tag': 'ldl_bimodal_drift_split', 'rule': '1-drift'}
        # One cell, not drifting. The over-merge finding is withdrawn and the
        # cluster continues down the flow, so a later rule can still fail it on
        # its own evidence -- this is a rescue from rule 1, not a pass. Rebinding
        # v restores the invariant asserted below ("from here, v is a CLEAN
        # verdict"), and is exactly the operation the three-arm A/B measured.
        v = 'not_overmerged'

    # Rule 0: independently fails ISI-VR (burst-correction didn't rescue it)
    # -> FAIL, no override exists for this. Checked before rules 7/8 (LDL
    # ambiguous/untested) since those have no protected finding of their own
    # to preserve -- resolving them via the simpler, more certain ISI-VR
    # criterion is a straightforward win, not a dilution of anything.
    # Confirmed 2026-07-31 on real data: whenever screen_pass=True, the
    # independent recheck already agrees (0/5905 confirmed_pass disagree) --
    # so this only ever changes outcomes for ambiguous (166/448) and
    # untested (191/1253), correctly resolving what used to be a
    # needlessly-flagged case into a confidently-resolved fail.
    if passes_isi_ind is False:
        return {'decision': 'FAIL', 'needs_inspection': False, 'tag': 'isi_vr_fail', 'rule': '0'}

    # Rule R: this cluster's spikes are already counted under another cluster.
    #
    # PLACED AFTER RULES 1 AND 0, BEFORE EVERYTHING ELSE. The same principle that
    # puts rule 1 first applies twice here. A cluster showing real overmerge
    # structure, or independently failing ISI-VR, has a positive finding ABOUT
    # ITSELF, and letting redundancy claim the tag would understate what those
    # tests catch. But every rule below this one either issues a PASS (2/3/5) or
    # defers to the screen (7/8), and neither is appropriate for a cluster that is
    # a copy: passing it double-counts a neuron, and deferring asks the screen a
    # question it has no channel to answer.
    #
    # NOT A FAILURE. Redundant is its own outcome (taxonomy.REDUNDANT) because
    # these spikes are real and correctly sorted -- they are simply already
    # attributed. decision stays 'FAIL' only because that field means "do not
    # emit as an independent unit", which is true; the taxonomy category carries
    # the distinction that matters to a user.
    #
    # ABSENT MEANS THE TEST DID NOT RUN, and that is a pipeline configuration
    # rather than a per-cluster untested default: records.load_records sets
    # redundant_of on EVERY record (None when the test ran and found nothing)
    # whenever the enrichment is enabled. A record built by hand without the key
    # is judged on its other channels, which is the pre-2026-08-24 behaviour.
    redundant_of = r.get('redundant_of')
    if redundant_of is not None:
        return {'decision': 'FAIL', 'needs_inspection': False,
                'tag': 'redundant_re_detection', 'rule': 'R',
                'redundant_of': redundant_of}

    # THE `ambiguous` BRANCH WAS DELETED (2026-09-08), not disabled.
    #
    # It handled the case where the LDL test's two statistics contradict each
    # other, and it became unreachable on 2026-09-03 when that conflict was
    # resolved in favour of the L-ratio: `_apply_rules` now remaps
    # `ambiguous` -> `not_overmerged` at its first line, so nothing arrives
    # here. Measured: the branch took 1,644 clusters on the 2026-08-31 run and
    # ZERO on both corpora after the fix -- 53,428 real clusters and 7,925
    # synthetic ones.
    #
    # It is deleted rather than kept as a guard because it could return a PASS
    # (`flagged_ambiguous_ldl` with screen_pass), and unreachable code that can pass
    # a cluster is the kind of thing that gets re-enabled by accident. Its
    # sub-branches -- `7-within`, `7-neighbor`, `7-neighbor-retrieved`,
    # `7-neighbor-all`, and their attribution reasoning -- go with it; the
    # argument survives in git history and in RULE_ID, which still decodes those
    # ids so archived runs can be read.
    #
    # Rule 8 (`untested`) is NOT affected and stays: unlike ambiguous, there is
    # no secondary signal that has already resolved that population.
    if v == 'untested':
        return {'decision': 'PASS' if screen_pass else 'FAIL', 'needs_inspection': True,
                'tag': 'flagged_untested_ldl', 'rule': '8'}

    # From here, v is a CLEAN verdict (unimodal / not_overmerged).
    # Rule 2: screen_pass=True and LDL clean -> PASS, no override needed.
    if screen_pass:
        return {'decision': 'PASS', 'needs_inspection': False, 'tag': 'confirmed_pass', 'rule': '2'}

    # screen_pass=False, LDL clean, ISI-VR already confirmed True or unknown
    # (rule 0 above already caught the False case) -- the screen failed this
    # cluster for some reason the within-cluster LDL test doesn't confirm.
    # Resolve WHY.
    # A RETRIEVAL NEEDS AN AFFIRMATIVE CLEAN VERDICT, not merely a clean-looking
    # string. `not_overmerged` with an unresolved basis means the LDL test
    # could not be adjudicated -- that is `+unsplittable`, a group the
    # dip could not split at all. Rules 3 and 5 turn a screen failure into a
    # PASS on the strength of "LDL says clean", so if the LDL never answered,
    # the retrieval has nothing to stand on. Same principle as rule 7's
    # attribution, applied to retrieval.
    #
    # Measured: 5 of 847 rule-3 retrievals rest on an unresolved verdict, and
    # all have lratio_passed_measured=False, so no channel cleared them --
    # failing is the correct outcome. This is what exposed 220523_034_np3 clu 124
    # as a PASS with zero affirmative channels. Applies to rule 3 ONLY; see the
    # note at rule 5 for why that retrieval is different.
    from unitassay.curate.bimodality import is_resolved
    _basis = r.get('dip_basis', _MISSING)
    assert _basis is not _MISSING, (
        'curate(): dip_basis is required to reach a retrieval rule -- rules 3/5 '
        'must know whether the LDL verdict was adjudicated. Enrich the record '
        '(records.load_records does this) or pass dip_basis explicitly. Absent '
        'must not be silently treated as either resolved or unresolved.')
    ldl_adjudicated = is_resolved(_basis)

    dom = r.get('dominant_source')

    if dom == 'noise':
        if passes_isi_ind is None:
            return {'decision': 'FAIL', 'needs_inspection': True,
                    'tag': 'pending_isi_check_within_cluster', 'rule': '3-pending'}
        # RULE 3 IS RETIRED. It used to PASS here when
        # `noise_ch_flagged and neighbor_never_fails and ldl_adjudicated`, on the
        # reasoning that the screen's noise verdict rests on a reference channel we
        # know was compromised, so the failure should not stand.
        #
        # THE REASONING WAS SOUND AND THE RULE WAS STILL WRONG. It retrieves on an
        # ARTEFACT EXPLANATION -- "the evidence against this cluster is unreliable"
        # -- which is not the same as evidence FOR the cluster. The same line was
        # drawn when rule 3 (but not rule 5) was gated on
        # ldl_adjudicated, because "LDL says clean" was its only positive evidence.
        # A clean within-cluster verdict says nothing about whether the cluster is
        # a fragment of a neighbour, and that is what it mostly turned out to be.
        #
        # MEASURED, both corpora, and they agree:
        #   synthetic  of 227 retrievals surviving the re-detection arm, 189 are
        #              over-split fragments and 35 are genuinely clean -- 15.4%
        #              precision. Retiring removes 157 of the 748 errors (21% of
        #              the budget) and moves accuracy 81.4% -> 85.3%, F1 64.4 ->
        #              68.5, at a cost of 3.7 points of recall.
        #   real       it passes 831 clusters (9.9% of all passes) that are 67%
        #              KS4 `mua` against 32% for passes overall -- 2.08x enriched
        #              in junk, on evidence that owes nothing to MEArec.
        #
        # NARROWINGS WERE TRIED AND ALL LOSE TO RETIRING: an extra "max overlap
        # with a neighbour <= t" condition separates the correct from the wrong
        # rescues at AUC 0.576 (p=0.16, i.e. not at all), and every threshold from
        # 2% to 15% scores below simple removal.
        #
        # THE ONE THAT LOOKED LIKE A FIX AND IS A TRAP: n_spikes separates them at
        # AUC 0.845 (correct rescues median 5,904 spikes, wrong ones 1,230). That
        # is an artefact of how MEArec is BUILT -- planted clean units are ~5,828
        # spikes and fragments are necessarily smaller. The real rule-3 population
        # has a median of 1,169 against 1,560 for all passes, i.e. the opposite
        # end of the distribution, so a spike-count condition would not have
        # transferred. Do not reintroduce it.
        #
        # What is NOT claimed: that the screen's noise verdict is sound. It is
        # measured at 3.9% detection on ground-truth noise clusters. The claim is only that this override
        # is empirically worse than the verdict it overrides.
        return {'decision': 'FAIL', 'needs_inspection': False,
                'tag': 'confirmed_fail_within_cluster', 'rule': '4'}

    if dom == 'other_cluster':
        if passes_isi_ind is None:
            return {'decision': 'FAIL', 'needs_inspection': True,
                    'tag': 'pending_isi_check_neighbor', 'rule': '5-pending-isi'}
        between_ok = r.get('between_cluster_confirmed_separated')
        # NOT gated on ldl_adjudicated, unlike rule 3. This retrieval rests
        # on an AFFIRMATIVE POSITIVE FINDING -- the between-cluster test says the
        # neighbour is separated -- which stands on its own evidence. Rule 3
        # retrieves on an ARTEFACT EXPLANATION (the reference was compromised),
        # so there "LDL says clean" is the only positive evidence and its
        # adjudication is load-bearing. Gating this one also sent clusters whose
        # between test HAD adjudicated into `pending_between_cluster_check`, a tag
        # asserting the opposite -- caught by taxonomy._neighbor_sub_type's guard.
        if between_ok is True:
            # ALL ACCUSERS, NOT JUST THE LOUDEST.
            # `between_cluster_confirmed_separated` adjudicates the DOMINANT
            # neighbour only -- the argmax of the per-neighbour L-ratio. 26% of
            # these clusters are accused by more than one neighbour, and rule 5
            # used to PASS them having tested one. Measured on all 983
            # retrievals: 36 (3.7%) have a QUIETER accuser that tests as
            # OVERLAPPING while the dominant one tests separated. They passed
            # only because of which neighbour happened to be loudest, which is
            # not a property of the cluster.
            #
            # This is the scope rule 3 has had all along (`neighbor_never_fails`
            # is explicitly all-neighbour, all-frame) and rule 5 did not.
            #
            # ABSENT IS NOT AFFIRMATIVE. None means some accuser could not be
            # adjudicated, so nothing cleared it -- that cannot support a
            # retrieval, and the cluster goes to the pending bucket rather than
            # passing. LOAD-BEARING, not defensive: this comment once said n=0
            # (true of the 50-session set it was measured on). On the 147 it
            # fired 231 times before between-cluster conflicts were decided by
            # the L-ratio (e17905f) and 13 times after.
            all_sep = r.get('between_all_neighbours_separated', _MISSING)
            assert all_sep is not _MISSING, (
                'curate(): between_all_neighbours_separated is required to reach '
                'rule 5\'s retrieval -- passing on the dominant neighbour alone '
                'leaves every other accuser untested. Enrich the record '
                '(records.load_records does this) or pass it explicitly. Absent '
                'must not be silently treated as "all clear".')
            if all_sep is False:
                return {'decision': 'FAIL', 'needs_inspection': False,
                        'tag': 'confirmed_fail_neighbor_real', 'rule': '5-all'}
            if all_sep is None:
                return {'decision': 'FAIL', 'needs_inspection': True,
                        'tag': 'pending_between_cluster_check', 'rule': '5-all-pending'}
            # THE TIME-DOMAIN GATE (2026-08-31). Everything above this point is
            # FEATURE SPACE, and feature space cannot answer the question rule 5
            # asks. Scored against per-pair ground truth -- two clusters sharing
            # a planted owner ARE one neuron split in two -- the dip is AUC
            # 0.496, the L-ratio 0.503, and a cross-validated linear discriminant
            # 0.503, which is the CEILING for that family. The cause is
            # circularity: the dip asks whether two clouds are separable in PC
            # space and KS4 guaranteed that when it drew the boundary.
            #
            # Spike timing is not party to the sorter's split, so it can
            # corroborate independently. `ccg_z_peak_trough` is a Poisson
            # deviation of the +-2 ms coincidence peak against the 5-20 ms flank
            # rate, two-sided: excess means the pair is one neuron re-detected
            # (75.6% of one-neuron pairs), a trough means a true refractory
            # partition (10.1%).
            #
            # REQUIRED, NOT OPTIONAL, and for the reason the rule-change skill
            # names: a retrieval that passes because the statistic was absent is
            # an untested default returning a PASS. 99.7% of the population has
            # one (13 of 4,658 do not), so the pending branch is negligible
            # rather than a hiding place.
            # THE TIME-DOMAIN GATE, OVER EVERY ACCUSER.
            #
            # Everything above this point is FEATURE SPACE, and feature space
            # cannot answer the question rule 5 asks: scored against per-pair
            # ground truth the dip is AUC 0.496, the L-ratio 0.503, and a
            # cross-validated LDA 0.503 -- the ceiling for that family. The cause
            # is circularity: the dip asks whether two clouds are separable in PC
            # space and KS4 guaranteed that when it drew the boundary. Spike
            # timing is not party to the sorter's split, so it can corroborate
            # independently.
            #
            # THE FIRST VERSION OF THIS GATE TESTED ONE NEIGHBOUR: the rule tested ALL accusers with the test that
            # does not discriminate and only the DOMINANT accuser with the test
            # that does. 1,422 of 4,588 clusters reaching this branch (31.0%)
            # have accusers the gate never saw. The same scope error, fixed for
            # the feature-space side, had been reintroduced one line below it.
            #
            # EACH PAIR AGAINST ITS OWN THRESHOLD, not max-z. The threshold is a
            # curve in n_min, and two accusers of one cluster can differ greatly
            # in n_min, so "which z is larger" and "which pair exceeds its own
            # threshold" can disagree. k is passed so the curve carries the
            # multiplicity correction (see curate/constants).
            acc = r.get('ccg_accusers', _MISSING)
            assert acc is not _MISSING, (
                "curate(): ccg_accusers is required to reach rule 5's retrieval "
                '-- a list of {z, n_min} over EVERY accusing neighbour. The '
                'feature-space evidence behind this retrieval is measured at '
                'chance (AUC 0.50), so passing without the time-domain check '
                'rests on nothing, and passing having checked only the loudest '
                'accuser is the same scope error. Enrich the record or pass it '
                'explicitly; absent must not be read as "clean".')
            if not acc:
                return {'decision': 'FAIL', 'needs_inspection': True,
                        'tag': 'pending_ccg_check', 'rule': '5-ccg-pending'}
            usable = [x for x in acc
                      if x.get('z') is not None and x.get('n_min') is not None]
            if not usable:
                return {'decision': 'FAIL', 'needs_inspection': True,
                        'tag': 'pending_ccg_check', 'rule': '5-ccg-pending'}
            k = len(usable)
            # EXCESS: the same event detected twice (re-detection).
            if any(x['z'] > ccg_z_threshold(x['n_min'], k) for x in usable):
                return {'decision': 'FAIL', 'needs_inspection': False,
                        'tag': 'declined_coincidence_excess', 'rule': '5-ccg'}
            # DEFICIT: fewer coincidences than chance, i.e. one neuron's own
            # refractory period forbidding its two halves from co-firing. A
            # SEPARATE test, not the other tail of the first one -- see the
            # constants: a deficit is bounded at z >= -sqrt(lam) and the shipped
            # two-sided statistic fired on it at the null rate.
            a_def = ccg_deficit_alpha(k)
            for x in usable:
                p = poisson_cdf(x.get('obs'), x.get('lam'))
                if p is not None and p < a_def:
                    return {'decision': 'FAIL', 'needs_inspection': False,
                            'tag': 'declined_coincidence_deficit', 'rule': '5-ccg-deficit'}
            return {'decision': 'PASS', 'needs_inspection': False,
                    'tag': 'retrieved_neighbor_confirmed', 'rule': '5'}
        if between_ok is False:
            return {'decision': 'FAIL', 'needs_inspection': False,
                    'tag': 'confirmed_fail_neighbor_real', 'rule': '5'}
        return {'decision': 'FAIL', 'needs_inspection': True,
                'tag': 'pending_between_cluster_check', 'rule': '5-pending'}

    # dominant_source unclassified/None (unexplained_fail_other) -- genuinely
    # not attributable to noise or neighbor even after the ISI-VR check.
    if passes_isi_ind is None:
        return {'decision': 'FAIL', 'needs_inspection': True,
                'tag': 'pending_isi_check_other', 'rule': 'other-pending'}
    return {'decision': 'FAIL', 'needs_inspection': True,
            'tag': 'unexplained_other_needs_review', 'rule': 'other'}
