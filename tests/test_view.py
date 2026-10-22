"""`unitassay view`: the viewer shows what curate() decided, from the same numbers.

Runs both layers on tests/fixture/ (as test_fixture_end_to_end does), then asks
the viewer's data layer and HTTP API about the result. No data files, no
network beyond a socket on 127.0.0.1.

THE PROPERTY THAT MATTERS is agreement, not drawing: the clean-frame fraction
the viewer reports must reproduce curate()'s own layer-1 verdict
(`screen_pass` is that fraction against ISOLATED_T_RATIO), and the pair CCG
statistic must equal the one written to pairs.json. A viewer that disagreed
with the decision it displays would be worse than none.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / 'fixture'


@pytest.fixture(scope='module')
def out(tmp_path_factory):
    from unitassay.cli import run_session
    from unitassay.curate import summarise_pairs
    d = tmp_path_factory.mktemp('view') / 'fx'
    d.mkdir()
    rows, timing, pairs = run_session(FIXTURE, d)
    # what `unitassay run` writes, minus the commit stamp
    (d / 'curated.json').write_text(json.dumps(
        {'version': 'test', 'commit': None, 'ks_dir': str(FIXTURE),
         'timing': timing, 'rows': rows}, default=float))
    (d / 'pairs.json').write_text(json.dumps(
        {'ks_dir': str(FIXTURE), 'summary': summarise_pairs(pairs),
         'pairs': pairs}, default=float))
    return d


def test_one_session_and_batch_root_both_open(out):
    from unitassay.view import Run
    assert list(Run(out).dirs) == ['fx']
    assert list(Run(out.parent).dirs) == ['fx']


def test_empty_root_refuses(tmp_path):
    from unitassay.view import Run
    with pytest.raises(FileNotFoundError):
        Run(tmp_path)


def test_clean_fraction_reproduces_screen_pass(out):
    from unitassay.view import Run
    from unitassay.curate.constants import ISOLATED_T_RATIO
    s = Run(out).session('fx')
    for clu, r in s.rows.items():
        iso = s.cluster(clu)['isolation']
        assert iso is not None and iso['n_frames'] > 0
        # screen_pass is the L-ratio half AND the ISI half (curation.py)
        assert ((iso['clean_frac'] >= ISOLATED_T_RATIO)
                and r['passes_isi_independent']) == r['screen_pass'], clu


def test_spike_counts_and_isi_match_the_row(out):
    from unitassay.view import Run
    s = Run(out).session('fx')
    for clu, r in s.rows.items():
        c = s.cluster(clu)
        assert c['n_spikes'] == r['n_spikes']
        assert c['isi']['n_viol'] == r['_n_viol']
    # cluster 2 is built with 12% refractory violations; the ACG must show it
    acg = s.cluster(2)['acg']
    lag, n = np.array(acg['lag_ms']), np.array(acg['counts'])
    assert n[np.abs(lag) < 1].sum() > 0


def test_pair_ccg_equals_pairs_json(out):
    from unitassay.view import Run
    s = Run(out).session('fx')
    checked = 0
    for p in (s.pairs or {}).get('pairs', []):
        z = (p.get('evidence') or {}).get('z')
        if z is None:
            continue
        st = s.ccg(p['a'], p['b'])['stats']
        assert st is not None and st['z'] == pytest.approx(z)
        checked += 1
    # the fixture may have no CCG-tested pair; the statistic is still computed
    st = s.ccg(0, 1)
    assert st['counts'] and len(st['lag_ms']) == len(st['counts'])


def test_events_need_one_session_or_a_map(out, tmp_path):
    from unitassay.view import Run
    ev = tmp_path / 'ev.npy'
    np.save(ev, np.array([1.0, 2.0, 3.0]))
    s = Run(out, events=str(ev)).session('fx')
    p = s.psth(0)
    assert p['n_trials'] == 3 and len(p['rate_hz']) == len(p['t_ms'])
    # a batch root with a bare file would apply one recording's events to all
    (out.parent / 'fx2').mkdir()
    (out.parent / 'fx2' / 'curated.json').write_text((out / 'curated.json').read_text())
    try:
        with pytest.raises(ValueError):
            Run(out.parent, events=str(ev))
    finally:
        (out.parent / 'fx2' / 'curated.json').unlink()
        (out.parent / 'fx2').rmdir()


def test_http_api_and_page(out):
    from http.server import ThreadingHTTPServer
    from unitassay.view import Run
    from unitassay.view.server import _handler
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _handler(Run(out)))
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        get = lambda p: urllib.request.urlopen(f'http://127.0.0.1:{port}{p}', timeout=30)
        page = get('/').read().decode()
        assert '<title>UnitAssay Viewer</title>' in page
        ss = json.loads(get('/api/sessions').read())['sessions']
        assert ss[0]['name'] == 'fx' and ss[0]['n'] == 4
        ov = json.loads(get('/api/session/fx').read())
        assert {r['clu_id'] for r in ov['rows']} == {0, 1, 2, 3}
        assert all(r['peak_ch'] is not None for r in ov['rows'])
        c = json.loads(get('/api/cluster/fx/0').read())
        assert c['row']['decision'] == 'PASS'
        with pytest.raises(urllib.error.HTTPError) as e:
            get('/api/session/nope')
        assert e.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_viewer_writes_nothing(out):
    """Read-only by design: the output directory is byte-identical afterwards."""
    from unitassay.view import Run
    before = {p: p.stat().st_mtime_ns for p in out.rglob('*')}
    s = Run(out).session('fx')
    s.overview()
    for clu in s.rows:
        s.cluster(clu)
    s.ccg(0, 1)
    after = {p: p.stat().st_mtime_ns for p in out.rglob('*')}
    assert before == after
