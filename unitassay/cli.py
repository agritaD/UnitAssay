"""unitassay command line — screen, then curate, in one call.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).

    unitassay run   --ks-dir <sorter_output> --out <dir>
    unitassay batch --sessions <list.tsv> --out <root> [--n-jobs 6]
    unitassay view  <out or root> [--port 8765] [--events times.npy]

Runs `screen` over a sorter's output to produce per-cluster isolation files,
then `curate` over those, and writes one JSON with every cluster's verdict and
the time each stage took. `batch` does the same across a list of recordings,
one subdirectory each, resuming only results from this version -- see
`unitassay/batch.py`.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Isolation window, seconds. The value every published result was produced at.
WINDOW_SEC = 100
# NOT the screen layer's own default of 64: that is aggressive on a shared
# machine and gains little, since the stage is I/O-bound as often as CPU-bound.
N_THREADS = 8
STAGES = ('extract', 'isolation', 'curation', 'iso_units')


def _isolate(ks_dir, out_dir, window_sec, n_threads):
    """Stage 1, as a SUBPROCESS.

    Deliberately not in-process: the screen layer ships a module named `types`
    that shadows the stdlib one when its directory leads sys.path, and `python
    -m` avoids that. It also means one crashed recording cannot take a batch
    down with it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / '_isolate_config.yaml'
    cfg.write_text(f'n_threads: {n_threads}\nisolation:\n  window_sec: {window_sec}\n')
    cmd = [sys.executable, '-m', 'unitassay.screen', 'run',
           '--config', str(cfg), '--ks-dir', str(ks_dir),
           '--output-dir', str(out_dir)]
    for s in STAGES:
        cmd += ['--stages', s]
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads),
               MKL_NUM_THREADS=str(n_threads), OPENBLAS_NUM_THREADS=str(n_threads))
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'screen failed ({p.returncode}): '
                           f'{(p.stderr or p.stdout)[-700:]}')
    return time.perf_counter() - t0


def run_session(ks_dir, out_dir, window_sec=WINDOW_SEC, n_threads=N_THREADS,
                reuse_isolation=False):
    """Both layers over one recording. Returns (rows, timing).

    `reuse_isolation=False` is the default and is the point: existing isolation
    files are DELETED before the stage runs, so a timing cannot silently
    describe a cache.
    """
    from unitassay.curate import curate_session, pair_findings

    ks_dir, out_dir = Path(ks_dir), Path(out_dir)
    iso = out_dir / 'isolation'
    timing = {'window_sec': window_sec, 'n_threads': n_threads}

    have = iso.is_dir() and any(iso.glob('*.SortData.h5'))
    if have and reuse_isolation:
        timing.update(screen_sec=None, isolation_reused=True)
    else:
        if have:
            for f in iso.glob('*.SortData.h5'):
                f.unlink()
        timing.update(screen_sec=round(_isolate(ks_dir, iso, window_sec,
                                                 n_threads), 1),
                      isolation_reused=False)

    t0 = time.perf_counter()
    # the ABSOLUTE path, not just the name: isolation lives under --out, which
    # is not required to be anywhere near --ks-dir
    rows = curate_session(ks_dir, isolation_dir=str(iso))
    timing['curate_sec'] = round(time.perf_counter() - t0, 1)
    # THE PAIRWISE VIEW, derived from the rows -- no new statistic, no verdict
    # changed. curate() decides per cluster, but duplication and over-splitting
    # are properties of a pair, and a per-cluster output under-reports them by
    # construction. See curate/pairs.py.
    pairs = pair_findings(rows)
    timing['n_pairs'] = len(pairs)
    timing['n_clusters'] = len(rows)
    # sum the NAMED stages: a loop over every key ending in `_sec` also catches
    # `window_sec`, which is config, and inflates the total by 100
    timing['total_sec'] = round(sum(timing.get(k) or 0.0
                                    for k in ('screen_sec', 'curate_sec')), 1)
    return rows, timing, pairs


def main(argv=None):
    ap = argparse.ArgumentParser(prog='unitassay')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='screen then curate one recording')
    r.add_argument('--ks-dir', required=True, help='sorter output directory')
    r.add_argument('--out', required=True)
    r.add_argument('--window-sec', type=int, default=WINDOW_SEC)
    r.add_argument('--n-threads', type=int, default=N_THREADS)
    r.add_argument('--reuse-isolation', action='store_true',
                   help='reuse existing isolation files; a timing measured this '
                        'way describes a cache, not a run')
    b = sub.add_parser('batch', help='screen then curate many recordings')
    b.add_argument('--sessions', required=True,
                   help='TSV of "name<TAB>ks_dir", one recording per line')
    b.add_argument('--out', required=True, help='output root; one subdir per session')
    b.add_argument('--n-jobs', type=int, default=6,
                   help='concurrent recordings (each uses --n-threads threads)')
    b.add_argument('--window-sec', type=int, default=WINDOW_SEC)
    b.add_argument('--n-threads', type=int, default=N_THREADS)
    v = sub.add_parser('view', help='browse one run or a batch root in a web browser (read-only)')
    v.add_argument('out', help='the --out of `run` (one session) or of `batch` (many)')
    v.add_argument('--port', type=int, default=8765)
    v.add_argument('--host', default='127.0.0.1',
                   help='127.0.0.1 by default; reach it remotely with an SSH tunnel')
    v.add_argument('--events', default=None,
                   help='event times in SECONDS on the recording clock (.npy or one per '
                        'line) to add a raster/PSTH; for a batch root, a .tsv of '
                        '"name<TAB>events_path"')
    a = ap.parse_args(argv)

    if a.cmd == 'view':
        from unitassay.view.server import serve
        serve(a.out, host=a.host, port=a.port, events=a.events)
        return 0

    if a.cmd == 'batch':
        from unitassay.batch import read_sessions, run_batch
        # the list is validated BEFORE the pool starts: a bad manifest should
        # cost a second, not an hour of workers failing one at a time
        ss = read_sessions(a.sessions)
        summary = run_batch(ss, a.out, n_jobs=a.n_jobs,
                            window_sec=a.window_sec, n_threads=a.n_threads,
                            log=lambda m: print(m, flush=True))
        return 0 if summary['n_failed'] == 0 else 1

    from unitassay.curate import summarise_pairs

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, timing, pairs = run_session(a.ks_dir, out, a.window_sec, a.n_threads,
                                      reuse_isolation=a.reuse_isolation)
    from unitassay import __version__, code_commit
    (out / 'curated.json').write_text(json.dumps(
        {'version': __version__, 'commit': code_commit(), 'ks_dir': str(a.ks_dir),
         'timing': timing, 'rows': rows}, default=float))
    (out / 'pairs.json').write_text(json.dumps(
        {'ks_dir': str(a.ks_dir), 'summary': summarise_pairs(pairs),
         'pairs': pairs}, default=float))
    from collections import Counter
    c = Counter(r_['decision'] for r_ in rows)
    print(f"{timing['n_clusters']} clusters   "
          f"PASS {c['PASS']}  FAIL {c['FAIL']}")
    print(f"screen {timing['screen_sec']}s   curate {timing['curate_sec']}s   "
          f"total {timing['total_sec']}s")
    ps = summarise_pairs(pairs)
    print(f"{ps['n_pairs']} pairs   {ps['n_not_adjudicated']} measured but not "
          f"adjudicated   {ps['n_both_pass']} with both members PASS")
    print(f"-> {out / 'curated.json'}\n-> {out / 'pairs.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
