"""unitassay batch — both layers over many recordings, with a resume that cannot
serve a stale result.

    unitassay batch --sessions list.tsv --out <root> [--n-jobs 6]

THE SESSION LIST IS AN INPUT, NOT SOMETHING THIS PACKAGE KNOWS. One
`name<TAB>ks_dir` per line, `#` for comments. A corpus manifest -- which
recordings, which sorter arm, how a name maps to a path -- belongs to whoever
owns the corpus; a distributable tool that hardcodes one cannot be used on
another.

INCREMENTAL BY SESSION, because a long job on a shared box does get killed:
each session writes its own file before the next starts.

THE RESUME IS THE ONE PLACE A STALE RESULT COULD SURVIVE, so it is not a file
existence check. A stored `curated.json` is trusted only if it records THIS
version AND its isolation files are still beside it. On 2026-09-04 a whole
147-session run produced complete, plausible verdicts from isolation it had not
written, and nothing errored; a resume that trusts a filename would reintroduce
exactly that.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path


def read_sessions(path):
    """`name<TAB>ks_dir` per line. Raises on a path that is not a directory.

    Checked HERE rather than in the worker: a typo in a manifest should stop the
    batch before it spends an hour, not surface as one failed session in a log.
    """
    out, bad = [], []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t') if '\t' in line else line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f'{path}:{i}: expected "name<TAB>ks_dir", got {line!r}')
        name, ks = parts[0].strip(), Path(parts[1].strip())
        if not ks.is_dir():
            bad.append(f'  {name}: {ks}')
        out.append((name, ks))
    if bad:
        raise NotADirectoryError('session list names paths that are not '
                                 'directories:\n' + '\n'.join(bad))
    if not out:
        raise ValueError(f'{path}: no sessions')
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        dup = {n for n in names if names.count(n) > 1}
        raise ValueError(f'{path}: duplicate session names {sorted(dup)} would '
                         'write to the same output directory')
    return out


def _one(args):
    name, ks_dir, out_root, window_sec, n_threads = args
    from unitassay import __version__, code_commit
    from unitassay.cli import run_session
    commit = code_commit()

    dest = Path(out_root) / name
    done = dest / 'curated.json'
    if done.exists():
        try:
            d = json.loads(done.read_text())
            iso_ok = any((dest / 'isolation').glob('*.SortData.h5'))
            # resume only a result from THIS code: same version AND same commit
            if (d.get('version') == __version__ and d.get('commit') == commit
                    and commit is not None and not commit.endswith('+dirty') and iso_ok):
                return name, d['timing'], 'resumed'
        except Exception:
            pass
        done.unlink()            # unreadable, from another version, or isolation gone
    try:
        rows, timing, pairs = run_session(ks_dir, dest, window_sec, n_threads)
    except Exception as e:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'error.json').write_text(json.dumps(
            {'session': name, 'ks_dir': str(ks_dir),
             'error': f'{type(e).__name__}: {e}'}, indent=1))
        return name, None, f'FAILED {type(e).__name__}'
    (dest / 'error.json').unlink(missing_ok=True)      # a retry that succeeded
    done.write_text(json.dumps(
        {'version': __version__, 'commit': commit, 'session': name,
         'ks_dir': str(ks_dir), 'timing': timing, 'rows': rows}, default=float))
    from unitassay.curate import summarise_pairs
    (dest / 'pairs.json').write_text(json.dumps(
        {'session': name, 'summary': summarise_pairs(pairs), 'pairs': pairs},
        default=float))
    return name, timing, 'ok'


def run_batch(sessions, out_root, n_jobs=6, window_sec=None, n_threads=None,
              log=print):
    """Returns the summary dict that is also written to `out_root/timing.json`."""
    from unitassay import __version__
    from unitassay.cli import N_THREADS, WINDOW_SEC

    window_sec = WINDOW_SEC if window_sec is None else window_sec
    n_threads = N_THREADS if n_threads is None else n_threads
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work = [(n, k, out_root, window_sec, n_threads) for n, k in sessions]

    log(f'unitassay {__version__}: {len(work)} sessions, n_jobs={n_jobs}, '
        f'window_sec={window_sec}, n_threads={n_threads}')
    t0 = time.perf_counter()
    rec, failed = [], []
    with Pool(n_jobs) as pool:
        for i, (name, timing, status) in enumerate(
                pool.imap_unordered(_one, work), 1):
            if timing:
                rec.append(dict(session=name, **timing))
                log(f'  [{i}/{len(work)}] {name:<22}{timing["n_clusters"]:>5} clu  '
                    f'iso {timing.get("screen_sec")}s  cur {timing["curate_sec"]}s  '
                    f'{status}')
            else:
                failed.append(name)
                log(f'  [{i}/{len(work)}] {name:<22}{status}')

    wall = time.perf_counter() - t0
    iso = [r['screen_sec'] for r in rec if r.get('screen_sec')]
    cur = [r['curate_sec'] for r in rec if r.get('curate_sec')]
    med = lambda v: round(sorted(v)[len(v) // 2], 1) if v else None
    summary = dict(
        version=__version__, n_sessions=len(rec), n_failed=len(failed),
        failed=failed, n_clusters=sum(r['n_clusters'] for r in rec),
        n_resumed=sum(1 for r in rec if r.get('isolation_reused')),
        n_jobs=n_jobs, window_sec=window_sec, n_threads=n_threads,
        wall_sec=round(wall, 1),
        isolate_cpu_sec=round(sum(iso), 1), curate_cpu_sec=round(sum(cur), 1),
        isolate_median_sec=med(iso), curate_median_sec=med(cur),
        sessions=rec)
    (out_root / 'timing.json').write_text(json.dumps(summary, indent=1))
    log(f'\n{len(rec)}/{len(work)} sessions, {summary["n_clusters"]:,} clusters')
    log(f'  wall clock       {wall / 60:.1f} min at n_jobs={n_jobs}')
    log(f'  screen CPU       {sum(iso) / 60:.1f} min   median {med(iso)} s/session')
    log(f'  curate  CPU      {sum(cur) / 60:.1f} min   median {med(cur)} s/session')
    if failed:
        log(f'  {len(failed)} FAILED: {", ".join(failed[:8])}'
            f'{" ..." if len(failed) > 8 else ""}   (see <session>/error.json)')
    log(f'  wrote {out_root / "timing.json"}')
    return summary
