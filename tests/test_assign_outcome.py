"""assign_outcome: three decisions, and an error class that always agrees.

Unit-level, on hand-built rows, so each branch is exercised directly -- the
fixture has no FAIL that no finding names, which is the case that produced the
132 clean-labelled FAILs on the 147.
"""
import pytest

from unitassay.curate.constants import LOW_SPIKE_THRESHOLD
from unitassay.curate.taxonomy import DECISIONS, assign_outcome

HI, LO = LOW_SPIKE_THRESHOLD, LOW_SPIKE_THRESHOLD - 1


def row(decision, error_class, rule='2', n=HI, **kw):
    return dict(decision=decision, error_class=error_class, rule=rule,
                n_spikes=n, needs_inspection=True, **kw)


def test_under_the_floor_is_low_spike_whatever_was_decided():
    for d in ('PASS', 'FAIL'):
        r = assign_outcome(row(d, 'clean', n=LO))
        assert (r['decision'], r['error_class']) == ('LOW_SPIKE_COUNT', 'low_spike_count')


def test_at_the_floor_is_judged():
    assert assign_outcome(row('PASS', 'clean', n=HI))['decision'] == 'PASS'


def test_pass_is_always_clean():
    assert assign_outcome(row('PASS', 'oversplit'))['error_class'] == 'clean'


@pytest.mark.parametrize('rule,expect', [('5-all-pending', 'oversplit'),
                                         ('5-ccg-pending', 'oversplit'),
                                         ('R', 'duplicate'),
                                         ('1', 'mixed_cluster'),
                                         ('4', 'oversplit')])
def test_a_fail_no_finding_names_is_classed_by_its_stage(rule, expect):
    r = assign_outcome(row('FAIL', 'clean', rule=rule))
    assert (r['decision'], r['error_class']) == ('FAIL', expect)


def test_a_named_fail_keeps_its_class():
    assert assign_outcome(row('FAIL', 'duplicate', rule='R'))['error_class'] == 'duplicate'


def test_needs_inspection_is_not_emitted():
    assert 'needs_inspection' not in assign_outcome(row('FAIL', 'oversplit'))


def test_only_three_decisions_exist():
    assert DECISIONS == ('PASS', 'FAIL', 'LOW_SPIKE_COUNT')


def test_missing_spike_count_raises():
    with pytest.raises(ValueError):
        assign_outcome(dict(decision='PASS', error_class='clean', rule='2'))
