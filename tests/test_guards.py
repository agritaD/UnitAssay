"""The guards, each of which exists because something once went wrong silently.

Every test here corresponds to a real incident or a real defect. A guard with no
test is a comment: it survives only as long as nobody refactors past it.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
import json

import pytest

from unitassay.batch import read_sessions
from unitassay.curate.constants import ccg_deficit_alpha, ccg_z_threshold
from unitassay.curate.curation import curate_session
from unitassay.curate.decision_flow import curate


# --------------------------------------------------------------------------
# The isolation resolver. A bare NAME asks it to search, and on 2026-09-04 a
# whole 147-session run was curated from stale files because a directory of the
# same name sat inside every sorter directory. It did not fail; it produced a
# complete, plausible set of verdicts from the wrong data.
# --------------------------------------------------------------------------

def test_ambiguous_isolation_name_refuses_rather_than_preferring_one(tmp_path):
    ks = tmp_path / 'sorter'; (ks / 'iso').mkdir(parents=True)
    (tmp_path / 'iso').mkdir()                 # the same name BESIDE it too
    with pytest.raises(FileNotFoundError, match='BOTH inside'):
        curate_session(ks, isolation_dir='iso')


def test_missing_isolation_raises_rather_than_curating_nothing(tmp_path):
    ks = tmp_path / 'sorter'; ks.mkdir()
    with pytest.raises(FileNotFoundError, match='beside or inside'):
        curate_session(ks, isolation_dir='nowhere')


# --------------------------------------------------------------------------
# Rule 5 requires a time-domain statistic over EVERY accuser. Passing on the
# loudest alone leaves every other accuser untested; passing with none is an untested default
# returning a PASS, and the feature-space evidence behind the retrieval is
# measured at chance (AUC 0.50), so it rests on nothing.
# --------------------------------------------------------------------------

def _retrieval_record(**over):
    r = {'ldl_verdict': 'not_overmerged', 'screen_pass': False,
         'dominant_source': 'other_cluster', 'passes_isi_independent': True,
         'between_cluster_confirmed_separated': True,
         'between_all_neighbours_separated': True, 'n_spikes': 1000,
         'redundant_of': None, 'isi_rate': 0.0, 'dip_basis': 'full+atomic'}
    r.update(over)
    return r


def test_a_retrieval_without_dip_basis_raises():
    """Rules 3 and 5 must know whether the LDL verdict was ADJUDICATED.

    Absent must not be read as either resolved or unresolved -- a retrieval
    resting on 'LDL says clean' when LDL never ran is an untested default
    returning a PASS.
    """
    r = _retrieval_record(ccg_accusers=[])
    del r['dip_basis']
    with pytest.raises(AssertionError, match='dip_basis is required'):
        curate(r)


def test_a_clean_accuser_retrieves():
    """The positive control for the four negative ones below."""
    out = curate(_retrieval_record(
        ccg_accusers=[{'clu': 1, 'z': 1.0, 'n_min': 500, 'obs': 2.0, 'lam': 2.0}]))
    assert out['decision'] == 'PASS'
    assert out['tag'] == 'retrieved_neighbor_confirmed'


def test_rule5_retrieval_without_accusers_raises():
    r = _retrieval_record()                    # no ccg_accusers key at all
    with pytest.raises(AssertionError, match='ccg_accusers is required'):
        curate(r)


def test_rule5_retrieval_without_all_neighbours_raises():
    r = _retrieval_record(ccg_accusers=[])
    del r['between_all_neighbours_separated']
    with pytest.raises(AssertionError, match='between_all_neighbours_separated'):
        curate(r)


def test_empty_accuser_list_pends_rather_than_passes():
    """Nothing cleared it, so it must not be read as clean."""
    out = curate(_retrieval_record(ccg_accusers=[]))
    assert out['decision'] == 'FAIL'
    assert out['tag'] == 'pending_ccg_check'


def test_accusers_with_no_statistic_pend_rather_than_pass():
    """A statistic that could not be measured is not a clean one."""
    out = curate(_retrieval_record(ccg_accusers=[{'clu': 9, 'z': None, 'n_min': None}]))
    assert out['tag'] == 'pending_ccg_check'


def test_a_quiet_accuser_over_its_own_threshold_declines():
    """EACH PAIR AGAINST ITS OWN THRESHOLD, not max-z.

    The threshold is a curve in n_min, so the loudest z and the pair that
    exceeds its own threshold need not be the same pair. Here the larger z is
    comfortably under its threshold and the smaller one is over its own; a
    max-z gate would pass this cluster.
    """
    # THE THRESHOLD FALLS WITH SPIKE COUNT -- T(200,2)=37.6, T(20000,2)=4.7 --
    # so a small pair needs an enormous z to be declined and a large one very
    # little. That is what lets the two orderings come apart here.
    loud_but_ok = {'clu': 1, 'z': 30.0, 'n_min': 200, 'obs': 90.0, 'lam': 5.0}
    quiet_but_over = {'clu': 2, 'z': 5.0, 'n_min': 20000, 'obs': 50.0, 'lam': 40.0}
    k = 2
    assert loud_but_ok['z'] <= ccg_z_threshold(loud_but_ok['n_min'], k)
    assert quiet_but_over['z'] > ccg_z_threshold(quiet_but_over['n_min'], k)
    assert loud_but_ok['z'] > quiet_but_over['z'], 'the max-z pair must be the CLEAN one'
    out = curate(_retrieval_record(ccg_accusers=[loud_but_ok, quiet_but_over]))
    assert out['decision'] == 'FAIL'
    assert out['tag'] == 'declined_coincidence_excess'


def test_threshold_rises_with_multiplicity():
    """k is the multiplicity correction: more accusers, a harder threshold."""
    assert ccg_z_threshold(500, 3) > ccg_z_threshold(500, 1)
    assert ccg_deficit_alpha(3) < ccg_deficit_alpha(1)


def test_threshold_falls_monotonically_with_spike_count():
    """Not a constant, and the DIRECTION is load-bearing.

    T(200,1)=21.9 against T(20000,1)=3.4: a small pair needs a huge excess to be
    declined. That is why declines concentrate on large clusters, and it is a
    power limit of the gate rather than a property of the clusters.
    """
    ns = [200, 305, 500, 1000, 5000, 20000]
    ts = [ccg_z_threshold(n, 1) for n in ns]
    assert ts == sorted(ts, reverse=True), ts
    assert ts[0] > 5 * ts[-1]


# --------------------------------------------------------------------------
# The batch's session list and resume.
# --------------------------------------------------------------------------

def test_session_list_rejects_a_path_that_is_not_a_directory(tmp_path):
    """A typo should cost a second, not an hour of workers failing one by one."""
    f = tmp_path / 's.tsv'; f.write_text('a\t/no/such/dir\n')
    with pytest.raises(NotADirectoryError):
        read_sessions(f)


def test_session_list_rejects_duplicate_names(tmp_path):
    """Two rows with one name would write to the same output directory."""
    d = tmp_path / 'ks'; d.mkdir()
    f = tmp_path / 's.tsv'; f.write_text(f'a\t{d}\na\t{d}\n')
    with pytest.raises(ValueError, match='duplicate'):
        read_sessions(f)


def test_session_list_rejects_a_malformed_line(tmp_path):
    f = tmp_path / 's.tsv'; f.write_text('only-a-name\n')
    with pytest.raises(ValueError, match='expected'):
        read_sessions(f)


def test_session_list_skips_comments_and_blanks(tmp_path):
    d = tmp_path / 'ks'; d.mkdir()
    f = tmp_path / 's.tsv'
    f.write_text(f'# a comment\n\na\t{d}\n')
    assert [n for n, _ in read_sessions(f)] == ['a']


def test_resume_refuses_a_result_from_another_version(tmp_path, monkeypatch):
    """A stored curated.json says nothing about WHICH isolation produced it."""
    from unitassay import batch
    dest = tmp_path / 'out' / 's1'; (dest / 'isolation').mkdir(parents=True)
    (dest / 'isolation' / '1.SortData.h5').write_bytes(b'')
    (dest / 'curated.json').write_text(json.dumps(
        {'version': '0.0.1-not-this-one', 'timing': {'n_clusters': 1}}))
    called = {}
    def _fake(ks_dir, out_dir, *a, **k):
        called['ran'] = True
        return [], {'n_clusters': 0, 'curate_sec': 0.0, 'screen_sec': 0.0}
    monkeypatch.setattr('unitassay.cli.run_session', _fake)
    batch._one(('s1', tmp_path, tmp_path / 'out', 100, 8))
    assert called.get('ran'), 'a result from another version was resumed'


def test_resume_refuses_when_the_isolation_is_gone(tmp_path, monkeypatch):
    """The verdicts are meaningless without the files they were computed from."""
    from unitassay import batch
    from unitassay import __version__
    dest = tmp_path / 'out' / 's1'; dest.mkdir(parents=True)
    (dest / 'curated.json').write_text(json.dumps(
        {'version': __version__, 'timing': {'n_clusters': 1}}))   # no isolation/
    called = {}
    def _fake(ks_dir, out_dir, *a, **k):
        called['ran'] = True
        return [], {'n_clusters': 0, 'curate_sec': 0.0, 'screen_sec': 0.0}
    monkeypatch.setattr('unitassay.cli.run_session', _fake)
    batch._one(('s1', tmp_path, tmp_path / 'out', 100, 8))
    assert called.get('ran'), 'a result whose isolation is gone was resumed'
