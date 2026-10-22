#!/usr/bin/env python3
"""curate/findings.py — every finding about a cluster, evaluated INDEPENDENTLY.

Every rule is evaluated, in parallel rather than in sequence.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
This is a DESCRIPTION layer. It does not decide anything and `curate()` does not
call it. `curate()` keeps its sequential rule table, unchanged and already
validated; this module answers the separate question "what is true about this
cluster", with no rule able to pre-empt another.

That split is deliberate and it is the whole reason this was cheap to build.
Measured before writing a line: the sequential order costs us **labels, never
decisions**. Rule R sits after rules 0 and 1, so 363 synthetic clusters that ARE
duplicates get described as `overmerge` or `isi_vr` -- and **0 of those 363 get
the wrong decision**, because every one of the competing rules also fails them.
Likewise 37% of synthetic and 23% of real clusters satisfy more than one rule's
condition, and the sequential flow reports exactly one, chosen by position.

So the harm the rule order does is entirely in the description, and a
description layer fixes all of it. Rewriting `_apply_rules`'s control flow would
put a stable, weeks-old tool at risk to fix a problem that is not in it.

WHY PARALLEL MATTERS FOR THE TAXONOMY
-------------------------------------
The taxonomy is OUTCOME-based, not mechanism-based, because different rules
evoking different mechanisms can arrive at one outcome. Measured, that holds -- after
duplicates are attributed, contamination (79.4%), refractory violations (68.2%)
and bimodality (48.7%) are ALL majority over-split. Three mechanisms, one
outcome.

It also dissolves a question that had no honest answer sequentially: an
over-split fragment is very often ALSO contaminated (median frac_other 0.148),
and a single `sub_type` had to pick one. Here it reports both.

THREE STATES, NOT TWO
---------------------
A predicate returns True, False, DECLINED or None:

    True      the finding is present
    False     the test ran and found nothing
    DECLINED  the test RAN and returned no verdict -- its statistics conflict
    None      the test COULD NOT RUN: its input was missing or insufficient

The first version had only True/False/None and put DECLINED and None together
under "untested", which made a rule that had run look like a rule that had not.
That is exactly the question above, and it had no good answer because the label
was wrong.

The distinction is real and it is in the tool already: `curate/bimodality`
names a within-cluster verdict `ambiguous` when the dip test and the L-ratio
CONTRADICT each other -- the cluster was measured twice and the two statistics
disagree -- and `untested` when the fit could not be attempted at all. 273 of
3,696 synthetic clusters are `ambiguous`; they were tested, thoroughly, and the
honest report is "we looked and cannot tell", not "we did not look".

ABSENT IS STILL NOT FALSE. A test that could not run must never read as
"clean" -- the defect class behind the L-ratio labelling bug and the rule-5 CCG
assert. It just must not read as "declined" either.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
from unitassay.curate.constants import (ABSENT_FRAME_FRAC_THRESH, ISOLATED_T_RATIO,
                             LOW_SPIKE_THRESHOLD, ccg_deficit_alpha,
                             ccg_z_threshold, poisson_cdf)
# rule 1's override thresholds live in decision_flow, not constants -- imported
# rather than copied so the adjudication here cannot drift from the rule there
from unitassay.curate.decision_flow import ISI_ONE_CELL_FLOOR, SPLIT_TIME_ORDERED_AUC

_MISSING = object()

# the test ran and returned no verdict -- distinct from None, which is "could
# not run". Never treat this as False.
DECLINED = 'DECLINED'


def _duplicate(r):
    """These spikes are already counted under another cluster (rule R)."""
    v = r.get('redundant_of', _MISSING)
    return None if v is _MISSING else bool(v)


def _refractory(r):
    """Refractory violations survive burst correction (rule 0)."""
    v = r.get('passes_isi_independent', _MISSING)
    return None if v is _MISSING or v is None else (v is False)


def _two_populations(r):
    """The LDL fit found two separable populations inside the cluster (rule 3).

    `ambiguous` means the fit RAN and its two statistics -- the dip test and the
    L-ratio -- contradicted each other. Since 2026-09-03 that is no longer left
    undecided: curate() gives PRECEDENCE TO THE L-RATIO, and this layer must
    apply the same precedence or the description and the decision disagree on
    the same cluster. On the 273 conflicted clusters the dip calls SEPARATED on
    272 and the L-ratio calls NOT separated; against ground truth the L-ratio is
    right 256/273 and the dip 17/273.

    That is a statement about the DIP being unreliable in this regime, not about
    the L-ratio being powerful -- only 16 of the 273 are truly over-merged, so
    the L-ratio's 93.8% is the base rate. Which 16 stays unknown, consistent
    with the within-cluster CCG (AUC 0.476, not powered) and the
    auto-correlogram (AUC 0.547, powered but not specific).

    `untested` / `insufficient` ARE None: the fit could not be attempted, which
    is a different thing from running and declining, and must not read as clean.
    """
    v = r.get('ldl_verdict', _MISSING)
    if v is _MISSING or v is None or v in ('untested', 'insufficient'):
        return None
    if v == 'ambiguous':
        return False          # L-ratio precedence: it said NOT separated
    return v == 'overmerged'


def _contaminated(r):
    """The screen's within-cluster arm condemns it (rule 4).

    NAMED FOR WHAT IT MEASURES, not for `dominant_source`'s value. The stored
    value is 'noise', but ground truth showed the intruders are usually other
    NEURONS: the clusters it condemns carry median frac_other 0.148 against
    frac_noise 0.015. Calling this finding `noise` would be a mislabel.
    """
    ds = r.get('dominant_source', _MISSING)
    ms = r.get('screen_pass', _MISSING)
    if ds is _MISSING or ms is _MISSING or ds is None:
        return None
    return ds == 'noise' and not ms


def _neighbour_overlap(r):
    """A neighbouring cluster is not separable from this one (rule 5's finding).

    This is the FINDING, not the verdict: `_neighbour_cleared` below is the
    adjudication that can cancel it. Keeping them apart is the one structural
    thing a parallel model must get right -- rule 5 turns a FAIL into a PASS, so
    collapsing the two would silently convert every retrieval into a rejection.
    """
    ds = r.get('dominant_source', _MISSING)
    ms = r.get('screen_pass', _MISSING)
    if ds is _MISSING or ms is _MISSING or ds is None:
        return None
    return ds in ('other_cluster', 'mixed') and not ms


def _neighbour_cleared(r):
    """ADJUDICATION, not a finding: every accuser tested separable.

    True clears `_neighbour_overlap`; False upholds it; None means the evidence
    was incomplete, which must NOT clear it.
    """
    allnb = r.get('between_all_neighbours_separated', _MISSING)
    one = r.get('between_cluster_confirmed_separated', _MISSING)
    v = allnb if allnb is not _MISSING and allnb is not None else (
        one if one is not _MISSING else None)
    return None if v is None else bool(v)


def _coincident(r):
    """The time domain says an accusing pair fires as one unit (rule 5's CCG).

    Excess OR deficit: a coincidence peak means the two are the same spikes, a
    refractory hole means they share a refractory period. Different mechanisms,
    same conclusion, so one finding -- the tags keep them countable apart.
    """
    acc = r.get('ccg_accusers', _MISSING)
    if acc is _MISSING or acc is None:
        return None
    usable = [x for x in acc if x.get('z') is not None and x.get('n_min') is not None]
    if not usable:
        return None if acc else False
    k = len(usable)
    if any(x['z'] > ccg_z_threshold(x['n_min'], k) for x in usable):
        return True
    a_def = ccg_deficit_alpha(k)
    for x in usable:
        p = poisson_cdf(x.get('obs'), x.get('lam'))
        if p is not None and p < a_def:
            return True
    return False


def _intermittent(r):
    """Absent from most frames, and clean where present (taxonomy's gate)."""
    a = r.get('absent_frame_frac', _MISSING)
    c = r.get('clean_frac_when_present', _MISSING)
    if a is _MISSING or c is _MISSING or a is None or c is None:
        return None
    return a > ABSENT_FRAME_FRAC_THRESH and c >= ISOLATED_T_RATIO


def _unattributed_ms_fail(r):
    """The screen failed it and NO attribution channel explains why.

    Its own `dominant_source` says the L-ratio passed, so the failure came from
    a criterion with nothing for us to attribute it to -- in practice frame
    COVERAGE, which is what the `intermittent` annotation then names. Without
    this finding the state reads as "no findings at all", i.e. indistinguishable
    from clean, which is precisely how seven genuinely clean but intermittent
    units sat in `unexplained` for eleven days.
    """
    ms = r.get('screen_pass', _MISSING)
    ds = r.get('dominant_source', _MISSING)
    if ms is _MISSING or ds is _MISSING or ds is None:
        return None
    return (ms is False) and ds == 'lratio_passed'


def _low_spike(r):
    """Below the floor at which the LDL test can run at all."""
    n = r.get('n_spikes', _MISSING)
    return None if n is _MISSING or n is None else (n < LOW_SPIKE_THRESHOLD)


PREDICATES = {
    'duplicate': _duplicate,
    'refractory': _refractory,
    'two_populations': _two_populations,
    'contaminated': _contaminated,
    'neighbour_overlap': _neighbour_overlap,
    'coincident': _coincident,
    'unattributed_ms_fail': _unattributed_ms_fail,
    'low_spike': _low_spike,
}

def _one_cell_not_drifting(r):
    """ADJUDICATION: two separable groups, but the timing says ONE cell.

    Mirrors rule 1's override exactly (decision_flow.py). The LDL establishes
    that the spikes form two separable groups; it does NOT establish that they
    are two NEURONS. Two quantities outside the feature space decide that:

      the RAW ISI rate      one neuron cannot fire twice inside its refractory
                            period, so a low rate says one cell (AUC 0.846)
      time-ordering of the  a drifting unit splits early-vs-late; a burst or
      split                 waveform-state split is interleaved

    When both say "one cell, not drifting", the over-merge finding is WITHDRAWN
    and the cluster is not carrying two populations at all.

    ABSENT IS NOT NEGATIVE: a missing isi_rate cannot claim "one cell", and a
    missing split_time_auc cannot claim "not drift", so either holds the finding.
    """
    if not bool(r.get('screen_pass')):
        return False
    isi = r.get('isi_rate')
    if isi is None or isi >= ISI_ONE_CELL_FLOOR:
        return False
    t_auc = r.get('split_time_auc')
    if t_auc is None or t_auc > SPLIT_TIME_ORDERED_AUC:
        return False
    return True


ADJUDICATIONS = {'neighbour_cleared': _neighbour_cleared,
                 'one_cell_not_drifting': _one_cell_not_drifting}

# NOT FINDINGS. An annotation is true or false about a cluster REGARDLESS of
# whether anything is wrong with it -- an intermittent unit can be perfectly
# clean, and measured, 3 real clusters are intermittent AND passing. Treating it
# as a finding made the layer fail them, which is the mechanism-as-outcome error
# in miniature. Annotations cross-cut outcomes, the way `high_drift` does.
ANNOTATIONS = {'intermittent': _intermittent}


def findings(r):
    """{'fired': frozenset, 'untested': frozenset, 'adjudications': dict}.

    `untested` is not decoration. A cluster with no findings AND nothing
    untested is clean on every channel we can read; a cluster with no findings
    and three untested channels has been asked almost nothing, and the two must
    never print the same.
    """
    fired, declined, not_run = set(), set(), set()
    for name, f in PREDICATES.items():
        v = f(r)
        if v is None:
            not_run.add(name)
        elif v is DECLINED:
            declined.add(name)
        elif v:
            fired.add(name)
    adj = {k: f(r) for k, f in ADJUDICATIONS.items()}
    ann = {k: f(r) for k, f in ANNOTATIONS.items()}
    ann['a_test_declined'] = None  # filled below, once `untested` is final
    # the adjudication cancels the finding it adjudicates -- never any other
    if 'neighbour_overlap' in fired and adj.get('neighbour_cleared') is True:
        fired.discard('neighbour_overlap')
    # NOT APPLICABLE IS NOT UNTESTED. The CCG tests ACCUSING NEIGHBOURS; a
    # cluster nothing accused has no pair to correlate, so a missing
    # `ccg_accusers` there means the test does not apply, not that we skipped it.
    # Treating the two alike put 6,243 of 16,917 real clusters into
    # "clean, a channel untested" -- 96% of that bucket -- and made an
    # enrichment's COVERAGE look like a property of the clusters.
    if 'coincident' in not_run and 'neighbour_overlap' not in fired:
        not_run.discard('coincident')
    # the same withdrawal rule 1 applies: two groups, but timing says one cell
    if 'two_populations' in fired and adj.get('one_cell_not_drifting') is True:
        fired.discard('two_populations')
    ann['a_test_declined'] = bool(declined)
    ann['a_test_could_not_run'] = bool(not_run)
    return {'fired': frozenset(fired), 'declined': frozenset(declined),
            'not_run': frozenset(not_run),
            # kept for callers that only ask "was anything unresolved"
            'untested': frozenset(declined | not_run),
            'adjudications': adj, 'annotations': ann}


# OUTCOME PRIORITY. The one place order is written down, and it is a REPORTING
# choice, not a semantic one -- every finding is still carried in `fired`.
#
# ORDERED BY MEASURED PURITY against ground truth, not by intuition. Each entry
# is (finding, outcome, how often that outcome is right when the finding fires),
# measured on the 28-recording corpus with duplicates attributed to rule R:
#
#   duplicate           -> duplicate     91.5%
#   coincident          -> oversplit     the pair fires as one unit
#   neighbour_overlap   -> oversplit     96.2%  (neighbor_indistinguishable)
#   contaminated        -> oversplit     79.4%
#   refractory          -> oversplit     68.2%
#   two_populations     -> mixed_cluster  cause not resolved
#
# `two_populations` is LAST and does not claim over-merge. The LDL test cannot
# separate "two neurons merged" from "one fragment that is itself bimodal", and
# neither can the two time-domain tests that were tried: the within-cluster CCG
# is not powered at these spike counts (AUC 0.476, needing ~1,900 spikes per
# sub-population against an observed median of 319) and the auto-correlogram is
# powered but not specific (AUC 0.547, because BOTH errors contaminate).
#
# NAMED `mixed_cluster`, over `two_separable_groups`, `bimodal_cause_unknown`
# and `split_or_merge_undetermined`. The last is confusing: over-split and
# over-merge are OPPOSITE
# errors, so a label that asks the reader to weigh one against the other reads
# as incoherent. `mixed_cluster` names the cluster's state instead -- it is not
# one clean unit -- and leaves the cause open.
#
# THE OVERCLAIM, STATED: 9 of the 28 are ground-truth CLEAN, so the name is
# wrong for a third of them. It is kept because all 28 FAIL either way and the
# actionable content -- do not use this as a single unit -- is right for all of
# them; and because the alternative names describe the EVIDENCE, which is the
# thing this outcome vocabulary exists to stop doing.
OUTCOME_PRIORITY = (
    ('low_spike', 'low_spike_count'),
    ('duplicate', 'duplicate'),
    ('coincident', 'oversplit'),
    ('neighbour_overlap', 'oversplit'),
    ('contaminated', 'oversplit'),
    ('refractory', 'oversplit'),
    ('two_populations', 'mixed_cluster'),
    ('unattributed_ms_fail', 'unattributed'),
)

# NOT AN OUTCOME. "a test declined to answer" is a statement about OUR EVIDENCE,
# not about what the cluster IS, so it rides along as an annotation the way
# `intermittent` and `high_drift` do. It was briefly an outcome
# (`clean_untested_channels`) and that was the mechanism-as-outcome error again,
# one level up: it split `clean` in two on a fact about the tool rather than a
# fact about the neuron.
#
# The distinction is still worth carrying -- the 83 synthetic clusters it marks
# are 61.4% truly clean against plain clean's 73.5% -- but as a caveat ON clean,
# not as a rival to it.

# Declared but essentially never assigned, and that is the point. Ground truth
# holds 23 `noise_contaminated_primary` clusters and the tool identifies ONE:
# there is no test for noise contamination, only for other-neuron
# contamination. A named outcome we almost never assign states the missing test; leaving
# it out of the vocabulary would hide it.
UNREACHED_OUTCOMES = ('noise_contaminated', 'noise_cluster')


def error_class(f):
    """One outcome for reporting, from a finding set. `fired` keeps the rest."""
    for name, out in OUTCOME_PRIORITY:
        if name in f['fired']:
            # `unattributed_ms_fail` is the screen failing on a cluster whose
            # L-ratio passed -- i.e. isolation NOT MAINTAINED OVER TIME, which is
            # exactly the paper's intermittent presence (Fig 4D). Reported as such
            # whether or not the coverage annotation also fired: no outcome may be
            # "unattributed".
            if out == 'unattributed':
                return 'intermittent_presence'
            return out
    return 'clean'
