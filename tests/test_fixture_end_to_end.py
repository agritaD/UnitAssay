"""Both layers over a real (tiny) sorter directory: does the pipeline RUN?

The unit tests exercise `curate()` on records. This one runs the thing a user
runs -- isolate reads the sorter's arrays and writes SortData files, curate
reads those and decides -- on `tests/fixture/`, in about a second.

WHAT IT CATCHES that the unit tests cannot: a break in the screen layer, in the
file formats between the two layers, in the subprocess call, or in the record
`curate_session` assembles. Every one of those sits outside `curate()` and none
of the other tests touches them.

THE GOLDEN VERDICTS ARE A LITERAL HERE, not a committed JSON, so that a change
to them shows up as a reviewable diff in this file rather than as a blob.

NOT A BENCHMARK. Four hand-made clusters say nothing about accuracy; that is
measured on the synthetic corpus with ground truth. See tests/make_fixture.py.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / 'fixture'

# clu_id: (n_spikes, decision, rule_id, error_class)
GOLDEN = {
    0: (700, 'PASS', '7', 'clean'),
    1: (650, 'PASS', '7', 'clean'),
    # 12% of its ISIs are inside the refractory window, by construction
    2: (600, 'FAIL', '2', 'oversplit'),
    # under the 240-spike floor: a verdict is returned, but named for what it is
    3: (120, 'PASS', '7', 'low_spike_count'),
}

REQUIRED = ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy',
            'amplitudes.npy', 'pc_features.npy', 'pc_feature_ind.npy',
            'templates.npy', 'channel_positions.npy', 'channel_map.npy',
            'whitening_mat_inv.npy', 'cluster_KSLabel.tsv', 'params.py')


def test_the_fixture_is_complete():
    """A partially committed fixture would fail the run test confusingly."""
    missing = [f for f in REQUIRED if not (FIXTURE / f).exists()]
    assert not missing, f'tests/fixture is missing {missing}'


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    from unitassay.cli import run_session
    out = tmp_path_factory.mktemp('run')
    rows, timing, pairs = run_session(FIXTURE, out)
    return {r['clu_id']: r for r in rows}, timing, out, pairs


def test_isolation_was_written(result):
    """isolate really ran: one SortData file per cluster, beside the verdicts."""
    _rows, timing, out, _pairs = result
    assert timing['isolation_reused'] is False
    assert len(list((out / 'isolation').glob('*.SortData.h5'))) == len(GOLDEN)


def test_every_cluster_is_curated(result):
    rows, timing, _out, _pairs = result
    assert set(rows) == set(GOLDEN)
    assert timing['n_clusters'] == len(GOLDEN)


@pytest.mark.parametrize('clu', sorted(GOLDEN))
def test_verdict_is_unchanged(result, clu):
    rows, _timing, _out, _pairs = result
    n, decision, rid, ec = GOLDEN[clu]
    r = rows[clu]
    assert r['n_spikes'] == n
    assert r['decision'] == decision
    assert r['rule_id'] == rid
    assert r['error_class'] == ec


def test_the_refractory_violator_is_the_one_that_fails(result):
    """The fixture is built so exactly one cluster fails, and for a known reason.

    If this ever passes, either the ISI test stopped firing or the fixture stopped
    containing violations -- and the two are worth telling apart, so the rate is
    asserted rather than just the verdict.
    """
    rows, _timing, _out, _pairs = result
    failed = [c for c, r in rows.items() if r['decision'] == 'FAIL']
    assert failed == [2]
    assert rows[2]['isi_rate'] > 0.05
    assert all(rows[c]['isi_rate'] < 0.01 for c in (0, 1, 3))


def test_the_row_does_not_carry_category(result):
    """`category` was Pass/Fail/Low-spike-count/Redundant.

    Pass/Fail duplicated `decision`; Low-spike-count and Redundant duplicated the
    error class. Verified equivalent on all 53,428 real clusters before removal:
    category == 'Pass' is exactly a PASS whose error class is `clean`.
    """
    rows, _timing, _out, _pairs = result
    for r in rows.values():
        assert 'category' not in r
        assert 'outcome' not in r, 'renamed to error_class'


def test_rows_carry_the_reader_facing_fields(result):
    """What a user reads off a row, on a real run rather than a synthetic record."""
    rows, _timing, _out, _pairs = result
    for r in rows.values():
        for f in ('decision', 'rule_id', 'rule_label', 'tag',
                  'error_class', 'sub_type', 'findings', 'n_spikes'):
            assert f in r, f


def test_the_pairwise_view_is_emitted(result):
    """curate() decides per cluster; duplication is a property of a pair.

    The fixture's four clusters are built far apart and share no spikes, so the
    honest expectation here is ZERO pairs -- which still exercises the code path
    and pins the contract. A fixture that started producing pairs would mean the
    clusters stopped being independent.
    """
    _rows, timing, _out, pairs = result
    from unitassay.curate import summarise_pairs
    assert timing['n_pairs'] == len(pairs)
    s = summarise_pairs(pairs)
    assert set(s) == {'n_pairs', 'by_channel', 'n_not_adjudicated', 'n_both_pass'}
    for p in pairs:
        assert p['a'] < p['b'], 'pairs must be canonically ordered'
        assert p['channel'] in ('re_detection', 'neighbour')
        assert p['action'] in ('none', 'review_pair', 'drop_a', 'drop_b')
        # EIGHT FIELDS, fixed. This tool already carries eight per-cluster
        # vocabularies; the pair record adds none, and the test says so, because
        # fields accrete one convenience at a time.
        assert set(p) == {'a', 'b', 'channel', 'both_pass', 'criterion_met',
                          'evidence', 'why', 'action'}


def test_a_run_emits_no_warnings(tmp_path):
    """A library must not spray warnings at its caller.

    An all-NaN slice is a REAL state here -- a cluster can be absent from a
    frame, so every neighbour L-ratio in it is NaN -- and numpy is right to warn
    about that in general. The fix is to handle the empty case, not to catch the
    warning and not to filter globally, which would hide the caller's own.
    """
    import warnings
    from unitassay.cli import run_session
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        run_session(FIXTURE, tmp_path / 'quiet')
    ours = [w for w in caught
            if 'unitassay' in str(getattr(w, 'filename', ''))]
    assert not ours, [f'{w.category.__name__}: {w.message}' for w in ours]
