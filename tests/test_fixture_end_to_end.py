"""Both layers over a real (tiny) sorter directory: does the pipeline RUN?

The unit tests exercise `curate()` on records. This one runs the thing a user
runs -- isolate reads the sorter's arrays and writes SortData files, curate
reads those and decides -- on `tests/fixture/`, in about a second.

WHAT IT CATCHES that the unit tests cannot: a break in the isolate layer, in the
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

# clu_id: (n_spikes, decision, rule_id, stage, outcome)
GOLDEN = {
    0: (700, 'PASS', '7', '7-clean', 'clean'),
    1: (650, 'PASS', '7', '7-clean', 'clean'),
    # 12% of its ISIs are inside the refractory window, by construction
    2: (600, 'FAIL', '2', '2-refractory', 'oversplit'),
    # under the 240-spike floor: a verdict is returned, but named for what it is
    3: (120, 'PASS', '7', '7-clean', 'low_spike_count'),
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
    rows, timing = run_session(FIXTURE, out)
    return {r['clu_id']: r for r in rows}, timing, out


def test_isolation_was_written(result):
    """isolate really ran: one SortData file per cluster, beside the verdicts."""
    _rows, timing, out = result
    assert timing['isolation_reused'] is False
    assert len(list((out / 'isolation').glob('*.SortData.h5'))) == len(GOLDEN)


def test_every_cluster_is_curated(result):
    rows, timing, _out = result
    assert set(rows) == set(GOLDEN)
    assert timing['n_clusters'] == len(GOLDEN)


@pytest.mark.parametrize('clu', sorted(GOLDEN))
def test_verdict_is_unchanged(result, clu):
    rows, _timing, _out = result
    n, decision, rid, stage, outcome = GOLDEN[clu]
    r = rows[clu]
    assert r['n_spikes'] == n
    assert r['decision'] == decision
    assert r['rule_id'] == rid
    assert r['stage'] == stage
    assert r['outcome'] == outcome


def test_the_refractory_violator_is_the_one_that_fails(result):
    """The fixture is built so exactly one cluster fails, and for a known reason.

    If this ever passes, either the ISI test stopped firing or the fixture stopped
    containing violations -- and the two are worth telling apart, so the rate is
    asserted rather than just the verdict.
    """
    rows, _timing, _out = result
    failed = [c for c, r in rows.items() if r['decision'] == 'FAIL']
    assert failed == [2]
    assert rows[2]['isi_rate'] > 0.05
    assert all(rows[c]['isi_rate'] < 0.01 for c in (0, 1, 3))


def test_rows_carry_the_reader_facing_fields(result):
    """What a user reads off a row, on a real run rather than a synthetic record."""
    rows, _timing, _out = result
    for r in rows.values():
        for f in ('decision', 'rule_id', 'rule_label', 'stage', 'tag',
                  'outcome', 'sub_type', 'findings', 'n_spikes'):
            assert f in r, f
