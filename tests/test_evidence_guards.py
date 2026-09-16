"""The evidence guards: a cluster must never PASS by exhaustion of objections.

Three guards route a cluster to a decision when a test could not return one,
rather than letting it fall through to clean:

    4b  the independent ISI check is absent          (0 fires on 61,353 clusters)
    5d  the neighbour evidence is incomplete         (3,403 fires, 6.4%)
    6   no verdict at all                            (0 fires)

4b and 6 have never fired on either corpus because the in-package producer
(`curate/curation.py`) casts the field with `bool()`, so it cannot emit None.
They exist for a CALLER-SUPPLIED record -- curate() takes a dict, and a caller
assembling rows themselves, or replaying a record older than the field, can
omit it. That path has no coverage from the corpora, which is what these tests
are for.
"""
import pytest

from unitassay.curate.decision_flow import RULE_ID, STAGE, curate


def _record(**over):
    """A record on the RETRIEVAL path: layer 1 failed it (`ms_pass=False`) but
    the within-cluster LDL verdict is clean, so curate() has to resolve WHY.

    `ms_pass=False` is load-bearing. The guard sits after `if ms_pass: return
    PASS`, so a record that layer 1 already passed never reaches it -- and it
    should not: `ms_pass` is computed as `passes_lratio and passes_isi`, so a
    True there already carries the ISI evidence the guard is checking for.
    """
    r = dict(clu_id=1, n_spikes=5000, ldl_verdict='clean', dip_basis='resolved',
             ms_pass=False, passes_isi_independent=True, dominant_source='noise',
             noise_ch_flagged=False, neighbor_never_fails=True, is_burst=False,
             isi_rate=0.0001, split_time_auc=0.5, ccg_accusers=[],
             overlap_pairs=[], redundant_of=None, absent_frame_frac=0.0,
             clean_frac_when_present=1.0, untested_channels=[],
             between_all_neighbours_separated=True,
             between_cluster_confirmed_separated=True, _dip_pval=0.9,
             _l_ratio=0.01, _n_viol=0)
    r.update(over)
    return r


def test_missing_isi_check_is_tagged_pending_not_passed():
    """4b: with dominant_source='noise' and the ISI field ABSENT, the cluster
    must FAIL as pending -- never PASS, and never be silently assumed."""
    r = _record()
    r.pop('passes_isi_independent')
    out = curate(r)
    assert out['decision'] == 'FAIL', 'a missing ISI check must not PASS'
    assert out['rule'] == '3-pending'
    assert out['needs_inspection'] is True
    assert RULE_ID[out['rule']][0] == '4b'
    assert STAGE[out['rule']] == '4-contamination'


def test_explicit_none_behaves_the_same_as_absent():
    """The contract says None means 'not yet computed', so an explicit None and
    an absent key must not diverge."""
    absent = _record(); absent.pop('passes_isi_independent')
    explicit = _record(passes_isi_independent=None)
    a, b = curate(absent), curate(explicit)
    assert a['decision'] == b['decision'] == 'FAIL'
    assert a['rule'] == b['rule'] == '3-pending'


def test_the_same_record_with_the_field_present_does_not_hit_the_guard():
    """Control: the guard must be reached by the MISSING field, not by the rest
    of the record. With the field present the cluster takes a real branch."""
    out = curate(_record(passes_isi_independent=True))
    assert out['rule'] != '3-pending'


def test_a_cluster_layer_1_passed_does_not_need_the_guard():
    """The invariant is not violated by rule 2 returning PASS without consulting
    the ISI field: `ms_pass` is `passes_lratio AND passes_isi`, so a True there
    already contains the evidence. Documented here so the shortcut is a
    recorded decision rather than an oversight."""
    r = _record(ms_pass=True)
    r.pop('passes_isi_independent')
    out = curate(r)
    assert out['decision'] == 'PASS' and out['rule'] == '2'


def test_every_stage_and_label_is_defined_for_every_tag():
    """No tag may reach a reader without a stage and a label -- that is how a
    verdict would appear unlabelled in a figure."""
    for tag in STAGE:
        assert tag in RULE_ID, f'{tag} has a stage but no rule label'
    for tag in RULE_ID:
        assert tag in STAGE, f'{tag} has a rule label but no stage'


@pytest.mark.parametrize('label', ['4b', '5d', '6'])
def test_the_guards_are_still_declared(label):
    """If a guard is ever deleted, this fails loudly rather than the invariant
    quietly weakening."""
    assert any(v[0] == label for v in RULE_ID.values()), \
        f'evidence guard {label} is no longer declared'
