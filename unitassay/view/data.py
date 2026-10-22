"""view/data.py — everything the viewer shows, as plain JSON-able dicts.

READ-ONLY. Nothing here writes a file. The viewer is for looking at what the
two layers measured and decided; it never edits a verdict.

NO STATISTIC IS COMPUTED HERE THAT THE PACKAGE ALREADY COMPUTES. The per-frame
L-ratios come from `process_sortdata_all_frames` -- the reader curate_session
itself uses -- and the clean/absent frames from `frame_matrix`, the function
`channel_fields` is built on. The ACG/CCG, ISI violations and PSTH call the
package's own `compute_ccg`, `accuser_stats`, `violation_stats` and `psth`. A
viewer that recomputed any of these would be a second implementation that can
disagree with the decision it is displaying.

What IS new here is display-only: subsampling PC clouds for plotting, binning
the firing rate over time, and picking each cluster's peak channel.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from unitassay.curate.ccg import (compute_ccg, accuser_stats, CCG_WINDOW_MS,
                                  CCG_BIN_MS, CCG_CENTRE_MS, CCG_FLANK_MS)
from unitassay.curate.constants import (L_RATIO_MS_THRESH, ISOLATED_T_RATIO,
                                        LOW_SPIKE_THRESHOLD, DIP_PVAL_THRESH,
                                        L_RATIO_THRESH)
from unitassay.curate.curation import _sample_rate
from unitassay.curate.isi_vr import (violation_stats, passes_isi_vr,
                                     REFRACTORY_MS, ISI_THRESH, NUM_VIOL_GUARD)
from unitassay.curate.sortdata import (sortdata_path, process_sortdata_all_frames,
                                       frame_matrix)

# Display caps. A 50,000-spike cluster drawn point for point is a slow page and
# says nothing more than 1,500 points do.
MAX_PC_POINTS = 1500
MAX_RASTER_TRIALS = 400
RATE_BIN_SEC = 5.0
ACG_WINDOW_MS = 50.0
ACG_BIN_MS = 0.5
PSTH_WINDOW_MS = (-300, 500)     # Betty's GUI window
PSTH_SMOOTH_MS = 10.0


def _f(x):
    """A JSON-safe float: NaN and inf become None."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _arr(a, nd=4):
    return [None if not np.isfinite(v) else round(float(v), nd)
            for v in np.asarray(a, dtype=float).ravel()]


def _subsample(a, n, seed):
    """At most n rows of a, the SAME rows every time for the same cluster, so
    stepping frames back and forth does not make the cloud flicker."""
    if a is None or len(a) <= n:
        return a
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[np.sort(idx)]


def _pts(a, seed):
    if a is None or np.ndim(a) != 2 or a.shape[0] == 0:
        return []
    a = _subsample(np.asarray(a, dtype=float), MAX_PC_POINTS, seed)
    return np.round(a[:, :3], 2).tolist()


def is_session_dir(p):
    return (Path(p) / 'curated.json').is_file()


def load_events(path):
    """Event times in SECONDS on the recording's clock (spike_times / fs).

    .npy, or a text file with one number per line. Returns a sorted float array.
    """
    p = Path(path)
    ev = np.load(p) if p.suffix == '.npy' else np.loadtxt(p, ndmin=1)
    ev = np.sort(np.asarray(ev, dtype=float).ravel())
    if ev.size == 0:
        raise ValueError(f'{p}: no events')
    return ev


def read_events_arg(arg, sessions):
    """`--events` -> {session_name: path}.

    A .tsv maps names to files ("name<TAB>path", the same shape as the batch
    session list). Anything else is ONE file, allowed only when exactly one
    session is being viewed -- applying one recording's events to another
    would draw a PSTH aligned to nothing, which looks like a flat unit.
    """
    if arg is None:
        return {}
    p = Path(arg)
    if p.suffix == '.tsv':
        out = {}
        for ln in p.read_text().splitlines():
            if not ln.strip() or ln.startswith('#'):
                continue
            name, path = ln.split('\t')[:2]
            out[name.strip()] = path.strip()
        unknown = sorted(set(out) - set(sessions))
        if unknown:
            raise ValueError(f'{p}: sessions not under the viewed root: {unknown}')
        return out
    if len(sessions) != 1:
        raise ValueError('one --events file needs exactly one session; for a '
                         'batch root pass a .tsv of "name<TAB>events_path"')
    return {sessions[0]: str(p)}


class Run:
    """A unitassay output: ONE session directory (it holds curated.json) or a
    batch root (one such directory per session)."""

    def __init__(self, root, events=None, cache=4):
        self.root = Path(root).resolve()
        if is_session_dir(self.root):
            self.dirs = OrderedDict([(self.root.name, self.root)])
        else:
            self.dirs = OrderedDict(
                (d.name, d) for d in sorted(self.root.iterdir())
                if d.is_dir() and is_session_dir(d))
        if not self.dirs:
            raise FileNotFoundError(
                f'no curated.json in {self.root} or its subdirectories -- '
                'point at the --out of `unitassay run` or `unitassay batch`')
        self.events = read_events_arg(events, list(self.dirs))
        self._cache, self._cap = OrderedDict(), cache

    def sessions(self):
        out = []
        for name, d in self.dirs.items():
            try:
                c = json.loads((d / 'curated.json').read_text())
            except Exception as e:      # one broken file must not hide the rest
                out.append(dict(name=name, error=str(e)))
                continue
            dec = [r.get('decision') for r in c.get('rows', [])]
            out.append(dict(name=name, n=len(dec), PASS=dec.count('PASS'),
                            FAIL=dec.count('FAIL'),
                            LOW_SPIKE_COUNT=dec.count('LOW_SPIKE_COUNT'),
                            commit=c.get('commit'), events=name in self.events))
        return out

    def session(self, name):
        if name not in self.dirs:
            raise KeyError(name)
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        s = Session(self.dirs[name], self.events.get(name))
        self._cache[name] = s
        while len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return s


class Session:
    """One recording's sorter arrays + isolation files + verdicts, lazily."""

    def __init__(self, sdir, events_path=None):
        self.dir = Path(sdir)
        c = json.loads((self.dir / 'curated.json').read_text())
        self.meta = {k: c.get(k) for k in ('version', 'commit', 'ks_dir', 'timing')}
        self.rows = {int(r['clu_id']): r for r in c['rows']}
        pf = self.dir / 'pairs.json'
        self.pairs = json.loads(pf.read_text()) if pf.is_file() else None
        self.ks = Path(c['ks_dir'])
        self.iso = self.dir / 'isolation'
        self.window_sec = (c.get('timing') or {}).get('window_sec')
        self.events_path = events_path
        self._spk = None
        self._events = None

    # ---- sorter arrays ----------------------------------------------------
    def _spikes(self):
        """Spike times (samples) grouped by cluster: one sort, then slices."""
        if self._spk is None:
            if not self.ks.is_dir():
                raise FileNotFoundError(
                    f'sorter directory {self.ks} (from curated.json) is not '
                    'reachable from this machine')
            fs = _sample_rate(self.ks)
            st = np.load(self.ks / 'spike_times.npy').ravel().astype(np.int64)
            sc = np.load(self.ks / 'spike_clusters.npy').ravel()
            order = np.argsort(sc, kind='stable')
            sc_sorted = sc[order]
            ids, starts = np.unique(sc_sorted, return_index=True)
            ends = np.r_[starts[1:], len(sc_sorted)]
            self._spk = dict(fs=fs, st=st[order], ids=ids, starts=starts,
                             ends=ends, duration=float(st.max()) / fs if len(st) else 0.0)
        return self._spk

    def times(self, clu):
        """Sorted spike times of one cluster, in samples."""
        s = self._spikes()
        i = np.searchsorted(s['ids'], clu)
        if i >= len(s['ids']) or s['ids'][i] != clu:
            return np.zeros(0, dtype=np.int64)
        return np.sort(s['st'][s['starts'][i]:s['ends'][i]])

    def _geometry(self):
        """Peak channel and its (x, y) for every template; display only."""
        tf, cf = self.ks / 'templates.npy', self.ks / 'channel_positions.npy'
        if not (tf.is_file() and cf.is_file()):
            return {}
        tmpl = np.load(tf, mmap_mode='r')
        pos = np.load(cf)
        ptp = np.ptp(np.asarray(tmpl), axis=1)          # (n_templates, n_ch)
        peak = np.argmax(ptp, axis=1)
        return {int(k): dict(ch=int(peak[k]), x=_f(pos[peak[k], 0]),
                             y=_f(pos[peak[k], 1]), amp=_f(ptp[k, peak[k]]))
                for k in range(len(peak))}

    # ---- views --------------------------------------------------------------
    def overview(self):
        geo = self._geometry()
        rows = []
        for clu, r in sorted(self.rows.items()):
            g = geo.get(clu, {})
            rows.append(dict(r, peak_ch=g.get('ch'), x=g.get('x'), y=g.get('y'),
                             template_amp=g.get('amp')))
        return dict(name=self.dir.name, dir=str(self.dir), **self.meta,
                    window_sec=self.window_sec, rows=rows,
                    pairs=self.pairs, has_events=self.events_path is not None,
                    thresholds=dict(l_ratio_ms=L_RATIO_MS_THRESH,
                                    isolated_t_ratio=ISOLATED_T_RATIO,
                                    low_spike=LOW_SPIKE_THRESHOLD,
                                    dip_pval=DIP_PVAL_THRESH,
                                    l_ratio_between=L_RATIO_THRESH,
                                    isi_ratio=ISI_THRESH,
                                    isi_guard=NUM_VIOL_GUARD,
                                    refractory_ms=REFRACTORY_MS))

    def isolation(self, clu):
        """Layer 1 for one cluster: per-frame PCs, waveforms and L-ratios.

        `lr_noise` / `lr_others` are exactly what curate() was handed. The PC
        clouds and waveforms are read from the same SortData.h5 for drawing.
        """
        f = sortdata_path(self.dir, clu, 'isolation')
        if not f.is_file():
            return None
        sd = process_sortdata_all_frames(f)
        out = dict(n_frames=0, other_clu=[], frames=[])
        if sd is None:
            return out
        _, absent, clean = frame_matrix(sd)
        with h5py.File(f, 'r') as h:
            n = int(h.attrs.get('n_frames', 0))
            fr0 = h['frame_0000'] if 'frame_0000' in h else None
            other = ([int(x) - 1 for x in fr0['other_clu'][:]]
                     if fr0 is not None and 'other_clu' in fr0 else [])
            good_bad = (_arr(fr0['other_good_bad'][:], 0)
                        if fr0 is not None and 'other_good_bad' in fr0 else [])
            frames = []
            for i in range(n):
                fk = f'frame_{i:04d}'
                if fk not in h:
                    frames.append(None)
                    continue
                fr = h[fk]
                g = lambda k: np.array(fr[k]) if k in fr else None
                others = ([np.array(fr['other'][k]) for k in sorted(fr['other'].keys())]
                          if 'other' in fr and hasattr(fr['other'], 'keys') else [])
                owf = g('other_clu_wf')
                frames.append(dict(
                    unit=_pts(g('unit'), clu * 1000 + i),
                    noise=_pts(g('noise'), clu * 1000 + i + 1),
                    others=[_pts(o, clu * 1000 + i + 2 + j) for j, o in enumerate(others)],
                    n_unit=int(g('unit').shape[0]) if g('unit') is not None and np.ndim(g('unit')) == 2 else 0,
                    clu_wf=_arr(g('clu_wf'), 3) if g('clu_wf') is not None else [],
                    noise_wf=_arr(g('noise_wf'), 3) if g('noise_wf') is not None else [],
                    other_wf=([_arr(w, 3) for w in np.atleast_2d(owf)]
                              if owf is not None and np.size(owf) else [])))
        return dict(n_frames=n, other_clu=other, other_good_bad=good_bad,
                    frames=frames,
                    lr_noise=_arr(sd['lr_noise']),
                    lr_others=[_arr(x) for x in sd['lr_others']],
                    absent=[bool(a) for a in absent],
                    clean=[bool(c) for c in clean],
                    clean_frac=_f(clean.mean()) if len(clean) else None)

    def cluster(self, clu):
        """Everything the cluster page draws, in one payload."""
        s = self._spikes()
        fs, dur = s['fs'], s['duration']
        t = self.times(clu)
        t_ms = t / fs * 1000.0
        n_viol, hill, n = violation_stats(t_ms)
        c, k = (compute_ccg(t, t, window_ms=ACG_WINDOW_MS, bin_ms=ACG_BIN_MS, fs=fs)
                if len(t) > 1 else (np.zeros(0), np.zeros(0)))
        edges = np.arange(0.0, dur + RATE_BIN_SEC, RATE_BIN_SEC)
        rate = (np.histogram(t / fs, edges)[0] / RATE_BIN_SEC if len(edges) > 1
                else np.zeros(0))
        my_pairs = []
        if self.pairs:
            my_pairs = [p for p in self.pairs.get('pairs', [])
                        if clu in (p.get('a'), p.get('b'))]
        return dict(clu_id=clu, row=self.rows.get(clu), n_spikes=int(n),
                    fs=fs, duration_sec=dur, window_sec=self.window_sec,
                    isi=dict(n_viol=n_viol, hill_ratio=_f(hill),
                             passes=bool(passes_isi_vr(t_ms)) if n > 1 else None),
                    acg=dict(lag_ms=_arr(c, 2), counts=[int(v) for v in k]),
                    rate=dict(bin_sec=RATE_BIN_SEC, hz=_arr(rate, 3)),
                    isolation=self.isolation(clu),
                    pairs=my_pairs)

    def ccg(self, a, b):
        """Cross-correlogram of a pair, with rule 5's statistic on the same
        spikes (the window and flanks are rule 5's, from curate/ccg.py)."""
        s = self._spikes()
        ta, tb = self.times(a), self.times(b)
        c, k = (compute_ccg(ta, tb, window_ms=CCG_WINDOW_MS, bin_ms=CCG_BIN_MS,
                            fs=s['fs']) if len(ta) and len(tb)
                else (np.zeros(0), np.zeros(0)))
        st = accuser_stats(ta, tb, fs=s['fs'])
        return dict(a=a, b=b, lag_ms=_arr(c, 2), counts=[int(v) for v in k],
                    centre_ms=CCG_CENTRE_MS, flank_ms=list(CCG_FLANK_MS),
                    stats={k_: _f(v) for k_, v in st.items()} if st else None)

    def psth(self, clu):
        """Event-aligned raster + PSTH, only if --events was given."""
        if self.events_path is None:
            return None
        from unitassay.screen.utils.psth import psth
        if self._events is None:
            self._events = load_events(self.events_path)
        s = self._spikes()
        t_ms = self.times(clu) / s['fs'] * 1000.0
        lo, hi = PSTH_WINDOW_MS
        trials = []
        for e in self._events * 1000.0:
            i0, i1 = np.searchsorted(t_ms, [e + lo, e + hi])
            trials.append(t_ms[i0:i1] - e)
        rate, n_tr = psth(trials, PSTH_WINDOW_MS, smoothing=PSTH_SMOOTH_MS)
        shown = trials[:MAX_RASTER_TRIALS]
        return dict(window_ms=list(PSTH_WINDOW_MS), n_trials=n_tr,
                    n_shown=len(shown),
                    raster=[_arr(x, 1) for x in shown],
                    t_ms=list(range(lo, hi + 1)), rate_hz=_arr(rate, 2))
