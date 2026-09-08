"""Characterisation tests for the rule table: every branch, no data files.

WHERE THESE RECORDS COME FROM. One minimal record per (rule, tag) branch, taken
from the 2026-09-08 run over 147 real sessions and cut down to only the keys
`curate()` actually reads. Each was checked to still reproduce the verdict the
full row got, so a fixture that silently lost a load-bearing key could not get
in here.

WHAT THEY ARE FOR. They pin the CURRENT behaviour of all 16 branches the real
corpus exercises. A change that moves any of them is not necessarily wrong --
but it must be deliberate, and it will show up here as a named branch rather
than as a shifted percentage in a 53,428-cluster summary three steps later.

WHAT THEY ARE NOT. Not ground truth: nothing here says a verdict is CORRECT,
only that it is what this code does. Accuracy is measured against the synthetic
corpus, not asserted in a unit test.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import pytest

from unitassay.curate.constants import LOW_SPIKE_THRESHOLD
from unitassay.curate.decision_flow import (RULE_ID, RULE_TITLE, STAGE,
                                            STAGE_QUESTION, curate, rule_id,
                                            rule_label, stage_of)


CASES = [
    # rule 0  ->  isi_vr_fail
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'full+atomic',
      'dominant_source': 'noise',
      'isi_rate': 0.008792345487457979,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 3868,
      'passes_isi_independent': False,
      'redundant_of': None,
      'split_time_auc': 0.5003785735903213},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '0',
      'stage': '2-refractory',
      'tag': 'isi_vr_fail'}),
    # rule 1  ->  confirmed_fail
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'recursive',
      'dominant_source': 'noise',
      'isi_rate': 0.0,
      'ldl_verdict': 'overmerged',
      'ms_pass': False,
      'n_spikes': 791,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5010740314537783},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '1',
      'stage': '3-composition',
      'tag': 'confirmed_fail'}),
    # rule 1  ->  ldl_bimodal_ms_passed
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'recursive',
      'dominant_source': 'lratio_passed',
      'isi_rate': 0.0009532888465204957,
      'ldl_verdict': 'overmerged',
      'ms_pass': True,
      'n_spikes': 2099,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5939706750192928},
     {'decision': 'FAIL',
      'needs_inspection': True,
      'rule': '1',
      'stage': '3-composition',
      'tag': 'ldl_bimodal_ms_passed'}),
    # rule 1-drift  ->  ldl_bimodal_drift_split
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'recursive',
      'dominant_source': 'lratio_passed',
      'isi_rate': 0.0,
      'ldl_verdict': 'overmerged',
      'ms_pass': True,
      'n_spikes': 20784,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.6464381743728965},
     {'decision': 'FAIL',
      'needs_inspection': True,
      'rule': '1-drift',
      'stage': '3-composition',
      'tag': 'ldl_bimodal_drift_split'}),
    # rule 2  ->  confirmed_pass
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'negligible_satellite+remainder_too_small',
      'dominant_source': 'lratio_passed',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': True,
      'n_spikes': 115,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5940713853599516},
     {'decision': 'PASS',
      'needs_inspection': False,
      'rule': '2',
      'stage': '7-clean',
      'tag': 'confirmed_pass'}),
    # rule 4  ->  confirmed_fail_within_cluster
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'full+atomic',
      'dominant_source': 'noise',
      'isi_rate': 0.007465781833264205,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 4823,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5038193864082051},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '4',
      'stage': '4-contamination',
      'tag': 'confirmed_fail_within_cluster'}),
    # rule 5  ->  confirmed_fail_neighbor_real
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': False,
      'ccg_accusers': [],
      'dip_basis': 'full+atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0033071517155849523,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 2420,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5000348336862236},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '5',
      'stage': '5-neighbour',
      'tag': 'confirmed_fail_neighbor_real'}),
    # rule 5  ->  retrieved_neighbor_confirmed
    ({'between_all_neighbours_separated': True,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [{'clu': 5,
                        'lam': 4.133333333333334,
                        'n_min': 305,
                        'obs': 5.0,
                        'z': 2.0636380861327486},
                       {'clu': 14,
                        'lam': 1.0666666666666667,
                        'n_min': 305,
                        'obs': 2.0,
                        'z': 2.37346441585572}],
      'dip_basis': 'lopsided_no_lratio+large_atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 305,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5310027519779842},
     {'decision': 'PASS',
      'needs_inspection': False,
      'rule': '5',
      'stage': '5-neighbour',
      'tag': 'retrieved_neighbor_confirmed'}),
    # rule 5-all  ->  confirmed_fail_neighbor_real
    ({'between_all_neighbours_separated': False,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [{'clu': 215,
                        'lam': 21.2,
                        'n_min': 347,
                        'obs': 19.0,
                        'z': 1.4435935245597855},
                       {'clu': 223,
                        'lam': 18.666666666666668,
                        'n_min': 347,
                        'obs': 33.0,
                        'z': 3.70970413401187},
                       {'clu': 221,
                        'lam': 1.8666666666666667,
                        'n_min': 337,
                        'obs': 1.0,
                        'z': 1.5871507864874146}],
      'dip_basis': 'lopsided_no_lratio+large_atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 347,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5518237990831174},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '5-all',
      'stage': '5-neighbour',
      'tag': 'confirmed_fail_neighbor_real'}),
    # rule 5-all-pending  ->  pending_between_cluster_check
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [{'clu': 296,
                        'lam': 75.46666666666667,
                        'n_min': 2309,
                        'obs': 58.0,
                        'z': 2.0946114049087043},
                       {'clu': 302,
                        'lam': 60.4,
                        'n_min': 2309,
                        'obs': 71.0,
                        'z': 2.3473953439210558}],
      'dip_basis': 'lopsided_no_lratio+large_atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0012998266897746968,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 2309,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5140688889389062},
     {'decision': 'FAIL',
      'needs_inspection': True,
      'rule': '5-all-pending',
      'stage': '5-neighbour',
      'tag': 'pending_between_cluster_check'}),
    # rule 5-ccg  ->  declined_coincidence_excess
    ({'between_all_neighbours_separated': True,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [{'clu': 136,
                        'lam': 343.73333333333335,
                        'n_min': 4583,
                        'obs': 338.0,
                        'z': 4.581816637877277}],
      'dip_basis': 'full+atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0030554343081623746,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 4583,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5159113398433661},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '5-ccg',
      'stage': '5-neighbour',
      'tag': 'declined_coincidence_excess'}),
    # rule 5-ccg-deficit  ->  declined_coincidence_deficit
    ({'between_all_neighbours_separated': True,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [{'clu': 42,
                        'lam': 91.46666666666667,
                        'n_min': 2428,
                        'obs': 65.0,
                        'z': 2.198351710476357}],
      'dip_basis': 'full+atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.00041203131437989287,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 2428,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5504839204340357},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '5-ccg-deficit',
      'stage': '5-neighbour',
      'tag': 'declined_coincidence_deficit'}),
    # rule 5-ccg-pending  ->  pending_ccg_check
    ({'between_all_neighbours_separated': True,
      'between_cluster_confirmed_separated': True,
      'ccg_accusers': [],
      'dip_basis': 'lopsided_no_lratio+remainder_too_small',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 234,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.6335744028051721},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': '5-ccg-pending',
      'stage': '5-neighbour',
      'tag': 'pending_ccg_check'}),
    # rule 5-pending  ->  pending_between_cluster_check
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'full+atomic',
      'dominant_source': 'other_cluster',
      'isi_rate': 0.001771479185119575,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 3388,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5260994774250114},
     {'decision': 'FAIL',
      'needs_inspection': True,
      'rule': '5-pending',
      'stage': '5-neighbour',
      'tag': 'pending_between_cluster_check'}),
    # rule R  ->  redundant_re_detection
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'negligible_satellite+remainder_too_small',
      'dominant_source': 'lratio_passed',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': True,
      'n_spikes': 88,
      'passes_isi_independent': True,
      'redundant_of': 384,
      'split_time_auc': 0.5160123966942148},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': 'R',
      'stage': '1-identity',
      'tag': 'redundant_re_detection'}),
    # rule other  ->  unexplained_other_needs_review
    ({'between_all_neighbours_separated': None,
      'between_cluster_confirmed_separated': None,
      'ccg_accusers': [],
      'dip_basis': 'negligible_satellite+remainder_too_small',
      'dominant_source': 'lratio_passed',
      'isi_rate': 0.0,
      'ldl_verdict': 'not_overmerged',
      'ms_pass': False,
      'n_spikes': 51,
      'passes_isi_independent': True,
      'redundant_of': None,
      'split_time_auc': 0.5},
     {'decision': 'FAIL',
      'needs_inspection': False,
      'rule': 'other',
      'stage': '8-fall-through',
      'tag': 'unexplained_other_needs_review'}),
]


@pytest.mark.parametrize('record,expected', CASES,
                         ids=[f"{e['rule']}-{e['tag']}" for _, e in CASES])
def test_branch_reproduces(record, expected):
    """Each branch still returns what it returned on the real corpus."""
    got = curate(dict(record))
    for k, v in expected.items():
        assert got[k] == v, f'{k}: {got[k]!r} != {v!r}'


def test_every_branch_the_corpus_exercises_is_covered():
    """16 is the number of (rule, tag) branches the 147 real sessions reach.

    If a rule change adds a branch, this fails until a case is added for it --
    which is the point: an untested branch should not reach a release quietly.
    """
    assert len(CASES) == 16
    assert len({(e['rule'], e['tag']) for _, e in CASES}) == 16


def test_curate_does_not_mutate_its_input():
    """`curate()` is called on live records all over the pipeline."""
    for record, _ in CASES:
        before = dict(record)
        curate(dict(record))
        assert record == before


def test_stage_agrees_with_the_rule_that_set_it():
    """One exit point sets `stage`, so it cannot disagree with `rule`."""
    for record, expected in CASES:
        got = curate(dict(record))
        assert got['stage'] == stage_of(got['rule'])


def test_rule_ids_are_canonical():
    """Legacy suffixed ids map onto the <number><letter> scheme."""
    assert rule_id('5-ccg') == '5b'
    assert rule_id('5-ccg-deficit') == '5c'
    assert rule_id('5-ccg-pending') == '5d'


def test_every_row_carries_the_reader_facing_id():
    """`rule` is the internal key; `rule_id` is what the README is indexed by.

    They are NOT interchangeable and their numbers collide: legacy '2' is a
    clean cluster while documented rule 2 is REFRACTORY. A row that carried only
    the legacy id made a reader who trusted the rule table misread every one.
    """
    for record, _ in CASES:
        got = curate(dict(record))
        assert got['rule_id'] == rule_id(got['rule'])
        assert got['rule_label'] == rule_label(got['rule'])
        assert got['rule_label'].startswith(got['rule_id'])


def test_the_collision_this_field_exists_for_is_real():
    """A regression guard for the reason `rule_id` was added.

    If these ever stop colliding, the legacy key has been renamed and the
    README's warning about it should go too.
    """
    assert rule_id('2') == '7' and RULE_TITLE['7'] == 'CLEAN'
    assert rule_id('0') == '2' and RULE_TITLE['2'] == 'REFRACTORY'
    assert rule_id('R') == '1' and RULE_TITLE['1'] == 'IDENTITY'


def test_the_two_naming_schemes_cannot_drift_apart():
    """RULE_ID and STAGE must describe the same rules, and agree on the number.

    Two hand-maintained tables over the same keys drift the moment a rule is
    added to one of them. The canonical number IS the stage number, so that is
    checkable rather than conventional.
    """
    assert set(RULE_ID) == set(STAGE)
    for legacy, (rid, _what) in RULE_ID.items():
        assert rid[0] == STAGE[legacy][0], legacy
        assert rid[0] in RULE_TITLE, legacy
        assert STAGE[legacy] in STAGE_QUESTION, legacy


def test_an_unknown_rule_raises_rather_than_getting_a_default():
    """A new rule must be placed in both tables deliberately."""
    with pytest.raises(KeyError):
        rule_id('9-invented')
    with pytest.raises(KeyError):
        stage_of('9-invented')


def test_below_the_floor_nothing_is_flagged_for_review():
    """A cluster under LOW_SPIKE_THRESHOLD cannot be flagged for inspection.

    Not cosmetic: `needs_inspection` is a request for human work, and below the
    floor no statistic can ever answer, so the request could never be closed.
    The DECISION must be untouched -- only the flag.
    """
    pending = [(r, e) for r, e in CASES if e['needs_inspection']]
    assert pending, 'no case with needs_inspection=True to exercise the override'
    for record, expected in pending:
        low = dict(record, n_spikes=LOW_SPIKE_THRESHOLD - 1)
        got = curate(low)
        assert got['needs_inspection'] is False
        assert got['decision'] == expected['decision'], 'the override moved a DECISION'


def test_missing_ldl_verdict_raises_rather_than_defaulting():
    """An absent verdict must not be read as 'not overmerged'."""
    record = dict(CASES[0][0])
    del record['ldl_verdict']
    with pytest.raises(KeyError):
        curate(record)
