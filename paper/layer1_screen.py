#!/usr/bin/env python3
"""Measured panels for the screen layer (layer 1).

Every function that produces a number is named for the quantity it regenerates
and takes the corpus root as an argument, so any of them can be run on its own:

    from paper.layer1_screen import threshold_sweep
    threshold_sweep(Path('.../unitassay_147_nb5'))['stability']['pct_pass']

`make_figure()` only draws. It measures nothing, so a figure can never disagree
with the numbers a reader regenerates.

WHAT THE SCREEN LAYER IS. A cheap first pass: for each cluster it picks the
channel the unit is largest on, picks a reference channel that is NOT a decayed
copy of that waveform, and asks per 100 s frame how separable the two are. A
cluster passes when the fraction of frames in which every L-ratio clears
`l_ratio_threshold` reaches `isolated_t_ratio`.

THE SHAPE-SIMILARITY GUARD IS THE POINT, not a detail. A Neuropixels unit is
seen across its whole footprint, so a nearby channel is usually a scaled copy of
the same waveform rather than background. Rejecting decay-like candidates is
what makes the reference mean anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHIPPED_L_RATIO = 0.2          # screen.config.CurationConfig.l_ratio_threshold
SHIPPED_T_RATIO = 0.6          # screen.config.CurationConfig.isolated_t_ratio
MIN_PC_COVERAGE = 0.1          # channel_info's min_pc_threshold


# ──────────────────────────────────────────────────────────────────────────
# data
# ──────────────────────────────────────────────────────────────────────────
def threshold_sweep(corpus, n_lratio_sessions=10,
                    l_grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0),
                    t_grid=None):
    """How the pass rate moves with each of the two shipped thresholds.

    The stability sweep is exact and free: `clean_frac_when_present` is the
    fraction of frames that cleared the L-ratio, so re-thresholding it needs no
    recomputation and covers the whole corpus.

    The L-ratio sweep is NOT free -- that stored fraction was computed AT the
    shipped threshold -- so the per-frame L-ratios are recomputed from the
    stored clouds with the package's own `_compute_l_ratio`, over a sample of
    sessions. It applies the SHIPPED criterion: the max over the noise arm and
    every flagged neighbour, per frame. Sweeping the noise arm alone describes
    a laxer test and gives 64.0% at 0.2 where the real criterion gives 53.4%.

    A PASS RATE IS YIELD, NOT CORRECTNESS. A threshold that passes everything
    maximises it. Read this beside `screen_vs_sorter`, never alone.
    """
    import h5py
    from unitassay.screen.curation.auto_curate import _compute_l_ratio

    corpus = Path(corpus)
    frac = []
    for p in sorted(corpus.glob('*/curated.json')):
        for r in json.loads(p.read_text())['rows']:
            v = r.get('clean_frac_when_present')
            if v is not None:
                frac.append(float(v))
    frac = np.asarray(frac)
    t_grid = np.round(np.arange(0, 1.001, 0.05), 2) if t_grid is None \
        else np.asarray(t_grid)

    passed = {g: 0 for g in l_grid}
    lratio_values = []          # per cluster: median over frames of the
    n_clu = 0                   # per-frame max across noise and neighbours
    for p in sorted(corpus.glob('*/curated.json'))[:n_lratio_sessions]:
        for h5p in sorted((p.parent / 'isolation').glob('*.SortData.h5')):
            try:
                with h5py.File(h5p, 'r') as g:
                    rows = []
                    for nm in sorted(k for k in g if k.startswith('frame_')):
                        fr = g[nm]
                        u = np.asarray(fr['unit']) if 'unit' in fr else None
                        nz = np.asarray(fr['noise']) if 'noise' in fr else None
                        if u is None or nz is None or not u.size or not nz.size:
                            rows.append(np.nan)
                            continue
                        vals = [_compute_l_ratio(u[:, :3], nz[:, :3])]
                        oth = fr.get('other')
                        if oth is not None:
                            items = ([oth[k] for k in oth] if hasattr(oth, 'keys')
                                     else list(np.asarray(oth)))
                            for o in items:
                                o = np.asarray(o)
                                if o.ndim == 2 and o.shape[0]:
                                    vals.append(_compute_l_ratio(u[:, :3], o[:, :3]))
                        rows.append(max(vals))
                    lr = np.asarray(rows, float)
            except Exception:
                continue
            ok = ~np.isnan(lr)
            if not ok.any():
                continue
            n_clu += 1
            lratio_values.append(float(np.median(lr[ok])))
            for g_ in l_grid:
                passed[g_] += int(np.mean(lr[ok] < g_) >= SHIPPED_T_RATIO)

    return dict(
        shipped=dict(l_ratio=SHIPPED_L_RATIO, t_ratio=SHIPPED_T_RATIO),
        stability=dict(t=t_grid.tolist(),
                       pct_pass=[float(100 * np.mean(frac >= t)) for t in t_grid],
                       n=int(frac.size)),
        lratio=dict(l=list(l_grid),
                    pct_pass=[100 * passed[g] / max(n_clu, 1) for g in l_grid],
                    n=n_clu, n_sessions=n_lratio_sessions,
                    values=lratio_values),
        stability_values=frac.tolist())


def screen_vs_sorter(results_json):
    """The screen's verdicts against the sorter's own label, on ground truth.

    Reads the output of the synthetic comparison rather than recomputing it,
    because that scorer owns several corrections that must not be duplicated
    (a clean single unit needs fmax >= 0.5 AND fpos <= 0.10; specificity is
    scored against noise-only negatives, with re-detections reported
    separately).

    DO NOT REDUCE THIS TO ONE BALANCED-ACCURACY NUMBER. `BTW_pair` -- two
    neurons on the same peak channel, built to defeat same-channel
    discrimination -- is 47% of the positives, so the aggregate reports corpus
    composition. Per category, or not at all.
    """
    w = json.loads(Path(results_json).read_text())
    t = w['tally']
    cats = {k: dict(n=v['n'], screen=v['ms'] / v['n'], sorter=v['ks'] / v['n'])
            for k, v in w['by_category'].items() if v['n']}
    return dict(n_recordings=w['n_recordings'], n_clean=w['n_clean'],
                by_category=cats, tally=t,
                duplicates=dict(total=t['KS4']['dup_total'],
                                screen_accepted=t['mind_snag']['dup_accept'],
                                sorter_accepted=t['KS4']['dup_accept']))


def cost_per_step(results_json):
    """Measured runtime of every pipeline step, per hour of recording."""
    return json.loads(Path(results_json).read_text())['cost']['per_step']


def channel_selection_agreement(corpus, n_sessions=12, min_spikes=100):
    """How often PC coverage would change the channel the screen picks.

    Regenerates the supplementary claim. The shipped `combined_score` uses
    `alpha = 1.0`, i.e. waveform energy alone, with PC coverage entering only as
    a fallback when the winner's coverage is below half. This measures what a
    genuine 50/50 blend would do instead.

    INDEX SPACES ARE NOT INTERCHANGEABLE. `templates.npy` here is dense over all
    channels while `pc_features` covers ten; the two are related through
    `pc_feature_ind`, and truncating by array position instead silently reads
    the template at channels 0-9 and drops all but a handful of clusters.
    """
    corpus = Path(corpus)
    agree = diff = fallback = tot = 0
    for p in sorted(corpus.glob('*/curated.json'))[:n_sessions]:
        ks = Path(json.loads(p.read_text())['ks_dir'])
        try:
            tm = np.load(ks / 'templates.npy')
            pcfi = np.load(ks / 'pc_feature_ind.npy')
            pcf = np.load(ks / 'pc_features.npy', mmap_mode='r')
            sc = np.load(ks / 'spike_clusters.npy').ravel().astype(np.int64)
        except Exception:
            continue
        for c in np.unique(sc):
            if c >= tm.shape[0] or c >= pcfi.shape[0]:
                continue
            idx = np.where(sc == c)[0]
            if idx.size < min_spikes:
                continue
            ch = pcfi[c].astype(int)
            ch = ch[ch < tm.shape[2]]
            if len(ch) < 3:
                continue
            ptp = np.sum(tm[c][:, ch] ** 2, axis=0)
            if ptp.max() <= 0:
                continue
            f = np.asarray(pcf[np.sort(idx[:1500])][:, :3, :len(ch)])
            nzr = 1.0 - np.mean(np.all(f == 0, axis=1), axis=0)
            pn = ptp / ptp.max()
            zn = nzr / (nzr.max() or 1)
            a_energy = int(np.argmax(pn))
            a_blend = int(np.argmax(0.5 * pn + 0.5 * zn))
            tot += 1
            agree += (a_energy == a_blend)
            diff += (a_energy != a_blend)
            fallback += (nzr[a_energy] < 0.5)
    return dict(n=tot, n_sessions=n_sessions,
                pct_same=100 * agree / max(tot, 1),
                pct_differ=100 * diff / max(tot, 1),
                pct_fallback_fires=100 * fallback / max(tot, 1))


# ──────────────────────────────────────────────────────────────────────────
# drawing — these only draw; they never measure
# ──────────────────────────────────────────────────────────────────────────
def _save(fig, out):
    """Write the panel, and make its marks individually editable.

    matplotlib writes repeated markers as one `<path>` in `<defs>` plus a
    `<use>` per point. That is vector, but Illustrator imports `<use>` as
    symbol instances that cannot be edited until expanded -- which is the same
    as a picture to whoever finishes the figure.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path as _P
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    if str(out).endswith('.svg'):
        from .svg_expand import expand_use
        expand_use(out)
    return _P(out)


def make_threshold_panel(sweep, out_svg):
    """Panel D: where each shipped threshold sits in the data it cuts.

    A SWEEP OF THE PASS RATE WAS THE WRONG PICTURE. It shows yield, and yield
    is maximised by a threshold that passes everything -- so the curve cannot
    argue for an operating point. The distribution can: the density AT the
    threshold IS the slope of that curve, so this carries the same sensitivity
    information and also shows whether the cut lands in a sparse region (robust)
    or on a peak (where a small change moves many clusters).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    from . import style
    style.apply()
    C, SZ = style.C, style.SZ
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.W_IN * 0.54, 1.75),
                               gridspec_kw=dict(wspace=0.40, left=0.095,
                                                right=0.985, top=0.76,
                                                bottom=0.28))
    thr_l = sweep['shipped']['l_ratio']
    thr_t = sweep['shipped']['t_ratio']

    v = _np.asarray(sweep['lratio'].get('values') or [])
    v = v[_np.isfinite(v) & (v > 0)]
    if v.size:
        bins = _np.logspace(_np.log10(max(v.min(), 1e-4)),
                            _np.log10(max(v.max(), 1e-3)), 44)
        a.hist(v, bins=bins, color=C['teal'], linewidth=0, rwidth=0.9)
        a.set_xscale('log')
    a.axvline(thr_l, color=C['warm'], lw=1.0, ls=(0, (3, 2)))
    a.text(thr_l, a.get_ylim()[1] * 1.02, f'{thr_l}', color=C['warm'],
           fontsize=SZ['tick'], fontweight='bold', va='bottom', ha='center')
    a.set_xlabel("cluster's L-ratio (median over frames)")
    a.set_ylabel('clusters')
    a.set_title('(i) the separation statistic', loc='left', fontsize=SZ['tick'])
    a.text(0.98, 0.96, f'n = {v.size:,}', transform=a.transAxes, ha='right',
           va='top', fontsize=SZ['tick'], color=C['neutral'])

    f = _np.asarray(sweep.get('stability_values') or [])
    b.hist(f, bins=_np.linspace(0, 1, 41), color=C['teal'], linewidth=0,
           rwidth=0.9)
    b.axvline(thr_t, color=C['warm'], lw=1.0, ls=(0, (3, 2)))
    b.text(thr_t, b.get_ylim()[1] * 1.02, f'{thr_t}', color=C['warm'],
           fontsize=SZ['tick'], fontweight='bold', va='bottom', ha='center')
    b.set_xlabel('fraction of frames the cluster is clean')
    b.set_ylabel('clusters')
    b.set_title('(ii) the stability statistic', loc='left', fontsize=SZ['tick'])
    b.text(0.98, 0.96, f'n = {f.size:,}', transform=b.transAxes, ha='right',
           va='top', fontsize=SZ['tick'], color=C['neutral'])
    for ax_, thr, data in ((a, thr_l, v), (b, thr_t, f)):
        if data.size:
            frac_pass = (data < thr).mean() if ax_ is a else (data >= thr).mean()
            ax_.text(0.98, 0.82, f'{100*frac_pass:.0f}% pass',
                     transform=ax_.transAxes, ha='right', va='top',
                     fontsize=SZ['tick'], color=C['warm'], fontweight='bold')
    _save(fig, out_svg)
    return out_svg


def make_performance_panel(perf, cost, out_svg):
    """Panel E: what the screen catches that the sorter's label does not, and
    what it costs.

    Deliberately NOT a single balanced-accuracy number. The corpus is 47%
    `BTW_pair`, built to defeat same-channel discrimination, so an aggregate
    reports composition. Per category, plus the duplicates the metric omits.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    C, SZ = style.C, style.SZ
    fig = plt.figure(figsize=(style.W_IN * 0.66, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.52,
                          left=0.275, right=0.985, top=0.86, bottom=0.30)

    cats = perf['by_category']
    order = sorted(cats, key=lambda k: cats[k]['screen'] - cats[k]['sorter'])
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(order))
    for i, k in enumerate(order):
        d = cats[k]
        delta = 100 * (d['screen'] - d['sorter'])
        ax.plot([100 * d['sorter'], 100 * d['screen']], [i, i], '-',
                lw=0.8, color=C['faint'], zorder=1)
        ax.plot(100 * d['sorter'], i, 'o', ms=2.8, color=C['neutral'], zorder=2)
        ax.plot(100 * d['screen'], i, 'o', ms=2.8,
                color=C['teal'] if delta > 0 else C['warm'], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k.replace('_', ' ')}  ({cats[k]['n']})"
                        for k in order], fontsize=SZ['tick'])
    ax.set_xlabel('% of known clean units kept'); ax.set_xlim(0, 105)
    ax.set_title('(i) the screen and the sorter catch different things',
                 loc='left', fontsize=SZ['tick'])
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls='', marker='o', ms=2.8,
                              color=C['neutral'], label="sorter's label"),
                       Line2D([], [], ls='', marker='o', ms=2.8,
                              color=C['teal'], label='screen')],
              loc='center left', bbox_to_anchor=(0.01, 0.42), ncol=1,
              handletextpad=0.3, labelspacing=0.4, fontsize=SZ['tick'])

    ax2 = fig.add_subplot(gs[1])
    D = perf['duplicates']
    v = [100 * D['sorter_accepted'] / D['total'],
         100 * D['screen_accepted'] / D['total']]
    ax2.bar([0, 1], v, 0.56, color=[C['neutral'], C['teal']], linewidth=0)
    for i, x in enumerate(v):
        ax2.text(i, x + 1.2, f'{x:.0f}%', ha='center', va='bottom',
                 fontsize=SZ['tick'], fontweight='bold',
                 color=[C['neutral'], C['teal']][i])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["sorter's\nlabel", 'screen'], fontsize=SZ['tick'])
    ax2.set_ylabel('% of re-detections accepted'); ax2.set_ylim(0, 40)
    ax2.set_title('(ii) and the metric omits this', loc='left',
                  fontsize=SZ['tick'])
    ax2.text(0.5, -0.30,
             f"{D['sorter_accepted'] - D['screen_accepted']:,} more of "
             f"{D['total']:,} rejected", transform=ax2.transAxes,
             ha='center', va='top', fontsize=SZ['tick'], color=C['neutral'])
    if 'screen' in cost:
        ax2.text(0.5, -0.48,
                 f"at {cost['screen']['ms_per_cluster']:.0f} ms per cluster",
                 transform=ax2.transAxes, ha='center', va='top',
                 fontsize=SZ['tick'], color=C['teal'], fontweight='bold')
    _save(fig, out_svg)
    return out_svg


def make_channel_agreement_panel(agreement, out_svg):
    """Supplementary: would weighting PC coverage change the channel chosen?

    Old manuscript Figure 4B/4C made this point on 16 sessions and called the
    alternative a "dual criterion". The shipped code hardcodes `alpha = 1.0`,
    so the combined score is waveform energy alone and PC coverage enters only
    as a fallback -- the label must say that, not imply a live weighting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    C, SZ = style.C, style.SZ
    fig, ax = plt.subplots(figsize=(style.W_IN * 0.44, 1.60),
                           gridspec_kw=dict(left=0.13, right=0.97, top=0.80,
                                            bottom=0.32))
    vals = [agreement['pct_same'], agreement['pct_differ'],
            agreement['pct_fallback_fires']]
    labs = ['same channel\nchosen', 'a 50/50 blend\nwould differ',
            'PC-coverage\nfallback fires']
    cols = [C['teal'], C['warm'], C['neutral']]
    ax.bar(range(3), vals, 0.56, color=cols, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=SZ['tick'], fontweight='bold', color=cols[i])
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=SZ['tick'])
    ax.set_ylabel('% of clusters'); ax.set_ylim(0, 112)
    ax.set_title('Waveform energy already decides the channel', loc='left',
                 fontsize=SZ['sub'])
    ax.text(0.99, 0.62, f"{agreement['n']:,} clusters\n"
            f"{agreement['n_sessions']} sessions", transform=ax.transAxes,
            ha='right', va='top', fontsize=SZ['tick'], color=C['neutral'],
            linespacing=1.4)
    _save(fig, out_svg)
    return out_svg


def dominance_by_criterion(corpus, n_sessions=None, min_spikes=100,
                           bins=((0, 50), (50, 90), (90, 100))):
    """Old manuscript Figure 4B/4C, on the full corpus, using the SHIPPED rule.

    "Dominance level" is a channel's PC coverage -- the fraction of the
    cluster's spikes that have non-zero PC features there. A channel the unit
    is barely present on is a poor choice however large its waveform looks.

    Two rules are compared, and only one of them is reconstructed here:

      energy    the naive choice -- highest (or lowest) waveform energy, full
                stop. This is the baseline the original figure compared against.
      unitassay `clus_channel_info()` ITSELF, called directly. Not a
                paraphrase: the shipped rule guards the unit channel on
                coverage and the reference channel on coverage AND
                shape-similarity, and an earlier version of this function
                approximated only the coverage half, which inverted the
                reference panel's conclusion.

    Returns per-session percentages so the spread can be drawn. The original
    panels were made on 16 sessions.
    """
    from unitassay.screen.utils.channel_info import clus_channel_info

    corpus = Path(corpus)
    out = {'unit': {'energy': [], 'unitassay': []},
           'reference': {'energy': [], 'unitassay': []},
           'bins': [list(b) for b in bins], 'sessions': []}
    files = sorted(corpus.glob('*/curated.json'))
    if n_sessions:
        files = files[:n_sessions]
    for p in files:
        ks = Path(json.loads(p.read_text())['ks_dir'])
        try:
            mx, mn, _flag = clus_channel_info(ks)          # THE SHIPPED RULE
            tm = np.load(ks / 'templates.npy')
            pcfi = np.load(ks / 'pc_feature_ind.npy')
            pcf = np.load(ks / 'pc_features.npy', mmap_mode='r')
            sc = np.load(ks / 'spike_clusters.npy').ravel().astype(np.int64)
        except Exception:
            continue
        got = {k: {'energy': [], 'unitassay': []} for k in ('unit', 'reference')}
        for c in np.unique(sc):
            if c >= tm.shape[0] or c >= pcfi.shape[0] or c >= len(mx):
                continue
            idx = np.where(sc == c)[0]
            if idx.size < min_spikes:
                continue
            ch = pcfi[c].astype(int)
            ch = ch[ch < tm.shape[2]]
            if len(ch) < 3:
                continue
            ptp = np.sum(tm[c][:, ch] ** 2, axis=0)
            if ptp.max() <= 0:
                continue
            f = np.asarray(pcf[np.sort(idx[:1500])][:, :3, :len(ch)])
            nzr = 1.0 - np.mean(np.all(f == 0, axis=1), axis=0)

            def coverage_of(channel):
                """PC coverage of the channel the shipped rule returned."""
                w = np.where(ch == int(channel))[0]
                return float(nzr[int(w[0])]) if w.size else np.nan

            got['unit']['energy'].append(float(nzr[int(np.argmax(ptp))]))
            got['reference']['energy'].append(float(nzr[int(np.argmin(ptp))]))
            got['unit']['unitassay'].append(coverage_of(mx[c]))
            got['reference']['unitassay'].append(coverage_of(mn[c]))
        if not got['unit']['energy']:
            continue
        out['sessions'].append(p.parent.name)
        for site in ('unit', 'reference'):
            for rule in ('energy', 'unitassay'):
                v = 100 * np.asarray(got[site][rule], float)
                v = v[np.isfinite(v)]
                out[site][rule].append(
                    [float(100 * np.mean((v >= lo) & (v <= hi)))
                     for lo, hi in bins] if v.size else [np.nan] * len(bins))
    out['n_sessions'] = len(out['sessions'])
    return out


EVIDENCE = {
    '4a reference channel': 'noise comparison',
    '5a feature space': 'neighbour comparison',
    '5b coincidence excess': 'neighbour comparison',
    '5c refractory deficit': 'neighbour comparison',
    '5d evidence incomplete': 'neighbour comparison',
    '8 unattributed': 'measured, not attributable',
}


def verdicts_resting_on_screen(corpus):
    """How much of the pipeline's output depends on what the screen measured.

    A far more useful claim about a first-pass screen than its own accuracy:
    the screen is a MEASUREMENT layer, and this is the share of final verdicts
    that could not have been reached without it.

    Also counts intermittent presence, the one class that only a per-frame
    measurement can reveal -- every other test is a single snapshot, in which
    a unit present for ten minutes and then silent looks like a sparse one.
    """
    from collections import Counter
    corpus = Path(corpus)
    kind, cls = Counter(), Counter()
    total = 0
    for p in sorted(corpus.glob('*/curated.json')):
        for r in json.loads(p.read_text())['rows']:
            total += 1
            lab = r.get('rule_label') or r.get('rule')
            kind[EVIDENCE.get(lab, 'decided without the screen')] += 1
            cls[r.get('error_class')] += 1
    on_screen = total - kind['decided without the screen']
    return dict(n_clusters=total, by_evidence=dict(kind),
                n_on_screen=on_screen, pct_on_screen=100 * on_screen / total,
                intermittent=cls.get('intermittent_presence', 0),
                error_class=dict(cls))


def make_contribution_panel(ev, cost, out_svg):
    """Panel E: what the screen buys, in the screen's own vocabulary.

    NOT 'how accurate is the screen'. Its verdict is a first pass that the
    later stage overrides where they disagree, so scoring it alone answers a
    question it was not built to win -- and on a corpus that is 47% adversarial
    pairs the aggregate reports composition anyway. What it IS for is the
    measurement, and this is the share of final verdicts that rest on it.

    No rule is named: the rules are introduced later, and at this point the
    reader knows only the three things the screen measures.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    C, SZ = style.C, style.SZ
    fig = plt.figure(figsize=(style.W_IN, 1.70))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.50], hspace=1.05,
                          left=0.022, right=0.985, top=0.76, bottom=0.10)

    ax = fig.add_subplot(gs[0])
    parts = [('noise comparison', ev['by_evidence'].get('noise comparison', 0), C['teal']),
             ('neighbour comparison', ev['by_evidence'].get('neighbour comparison', 0), '#4FA08C'),
             ('not attributable', ev['by_evidence'].get('measured, not attributable', 0), '#A8D0C6'),
             ('decided without it', ev['by_evidence'].get('decided without the screen', 0), C['rule'])]
    tot, left = ev['n_clusters'], 0.0
    for lab, n, col in parts:
        w = 100 * n / tot
        ax.barh(0, w, left=left, height=0.46, color=col, linewidth=0)
        if w > 8:
            ax.text(left + w / 2, 0, f'{w:.1f}%', ha='center', va='center',
                    fontsize=SZ['tick'],
                    color='white' if col != C['rule'] else C['grey'],
                    fontweight='bold')
        left += w
    ax.set_xlim(0, 100); ax.set_ylim(-1.15, 0.5)
    ax.set_yticks([]); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # fixed label columns: the third segment is 1.4% wide, so labels placed at
    # their own segment's start collide with the next one
    cols = [(0, 'left'), (24, 'left'), (48, 'left'), (100, 'right')]
    for (lab, n, col), (x, ha) in zip(parts, cols):
        ax.text(x, -0.52, f'{lab}\n{n:,}', fontsize=SZ['tick'],
                color=col if col != C['rule'] else C['grey'], va='top',
                linespacing=1.4, ha=ha)
    ax.set_title(f"{ev['pct_on_screen']:.0f}% of all "
                 f"{ev['n_clusters']:,} verdicts rest on what the screen measured",
                 loc='left', fontsize=SZ['sub'])

    ax2 = fig.add_subplot(gs[1]); ax2.axis('off')
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    ax2.text(0, 88, f"{ev['intermittent']:,} clusters are intermittently present — "
             "a class only a per-frame measurement can see", fontsize=SZ['lab'],
             color=C['ink'], va='top')
    ax2.text(0, 36, "every other test is a single snapshot, in which a unit present "
             "for ten minutes and then silent looks like a sparse one",
             fontsize=SZ['tick'], color=C['neutral'], va='top')
    if 'screen' in cost:
        ax2.text(100, 88, f"{cost['screen']['ms_per_cluster']:.0f} ms per cluster",
                 fontsize=SZ['lab'], color=C['teal'], fontweight='bold',
                 ha='right', va='top')
    _save(fig, out_svg)
    return out_svg


def make_dominance_panel(dom, out_svg):
    """Supplementary: old manuscript Figure 4B/4C, on the full corpus.

    Replicates the original's form deliberately -- a box per dominance band
    per rule, every session an overlaid dot, boxed axes, legend top left --
    so the two can be compared directly. Only the corpus changes: 16 sessions
    became 147.

    THE LEGEND SAYS WHAT SHIPS. The original labelled the second series "dual
    criterion". `alpha = 1.0` is hardcoded, so the combined score is waveform
    energy alone and coverage enters only as a fallback; the blend is shown as
    the alternative it is, not as the tool's behaviour.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    from matplotlib.patches import Patch
    from . import style
    style.apply()
    SZ = style.SZ
    DOT = {'energy': '#2E7D8A', 'unitassay': '#8B1A1A'}
    SWATCH = {'energy': '#ADD8E6', 'unitassay': '#F08080'}
    NAME = {'energy': 'Waveform energy alone', 'unitassay': 'unitassay'}
    labels = [f'{lo}-{hi}' for lo, hi in dom['bins']]

    fig, axes = plt.subplots(1, 2, figsize=(style.W_IN * 0.62, 2.05),
                             gridspec_kw=dict(wspace=0.34, left=0.095,
                                              right=0.99, top=0.84,
                                              bottom=0.22))
    titles = {'unit': 'Unit-channel selection',
              'reference': 'Noise-channel selection'}
    rng = _np.random.default_rng(0)
    for ax, site in zip(axes, ('unit', 'reference')):
        for j, rule in enumerate(('energy', 'unitassay')):
            arr = _np.asarray(dom[site][rule])          # sessions x bins
            for i in range(arr.shape[1]):
                x = i + (j - 0.5) * 0.30
                bp = ax.boxplot([arr[:, i]], positions=[x], widths=0.20,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='k', lw=0.7),
                                whiskerprops=dict(lw=0.5),
                                capprops=dict(lw=0.5))
                for b in bp['boxes']:
                    b.set(facecolor='white', edgecolor='k', linewidth=0.5)
                ax.scatter(x + rng.uniform(-0.075, 0.075, arr.shape[0]),
                           arr[:, i], s=1.6, color=DOT[rule], alpha=0.65,
                           linewidths=0, zorder=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=SZ['tick'])
        ax.set_xlim(-0.55, len(labels) - 0.45)
        ax.set_ylim(-4, 104)
        ax.set_xlabel('Dominance level (%)')
        ax.set_ylabel('Percentage of clusters (%)')
        for sp in ax.spines.values():          # the original is a boxed axes
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        ax.set_title(titles[site], loc='left', fontsize=SZ['sub'])
    axes[0].legend(handles=[Patch(facecolor=SWATCH[r], edgecolor='k',
                                  linewidth=0.5, label=NAME[r])
                            for r in ('energy', 'unitassay')],
                   loc='upper left', bbox_to_anchor=(0.02, 0.98),
                   fontsize=SZ['tick'], frameon=True, handlelength=1.1,
                   borderpad=0.3, labelspacing=0.3)
    axes[1].text(0.98, 0.04, f"{dom['n_sessions']} sessions",
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=SZ['tick'], color=style.C['neutral'])
    _save(fig, out_svg)
    return out_svg


def pass_rates_by_label(corpus, n_lratio_sessions=None,
                        l_grid=(0.1, 0.2, 0.4), t_grid=(0.3, 0.6, 0.9)):
    """Old manuscript Figure 5C/5D on the full corpus: how many clusters clear
    each threshold, split by the sorter's own label.

    Splitting by KS4's label is the point. If a criterion passed `good` and
    `mua` clusters at the same rate it would be measuring nothing the sorter
    had not already decided; the gap between the two is the evidence that it
    sees something else.

    L-RATIO SWEEP (5C) holds stability at its shipped value and moves the
    L-ratio cut, applying the SHIPPED criterion -- the max across the noise arm
    and every flagged neighbour, per frame. STABILITY SWEEP (5D) does the
    reverse, and is exact and free: `clean_frac_when_present` is the fraction
    of frames that cleared the L-ratio at its shipped value.

    Returns per-session percentages, so the panel can show the spread the way
    the original did.
    """
    import h5py
    from unitassay.screen.curation.auto_curate import _compute_l_ratio

    corpus = Path(corpus)
    out = {'l_grid': list(l_grid), 't_grid': list(t_grid),
           'lratio': {g: {'all': [], 'good': [], 'mua': []} for g in l_grid},
           'stability': {t: {'all': [], 'good': [], 'mua': []} for t in t_grid},
           'shipped': dict(l_ratio=SHIPPED_L_RATIO, t_ratio=SHIPPED_T_RATIO),
           'n_sessions_lratio': 0, 'n_sessions_stability': 0}

    def ks_labels(ks_dir):
        f = Path(ks_dir) / 'cluster_KSLabel.tsv'
        lab = {}
        if f.exists():
            for ln in f.read_text().splitlines()[1:]:
                b = ln.split('\t')
                if len(b) >= 2:
                    lab[int(b[0])] = b[1].strip().lower()
        return lab

    files = sorted(corpus.glob('*/curated.json'))
    for p in files:
        j = json.loads(p.read_text())
        lab = ks_labels(j['ks_dir'])
        rows = j['rows']
        groups = {'all': rows,
                  'good': [r for r in rows if lab.get(r['clu_id']) == 'good'],
                  'mua': [r for r in rows if lab.get(r['clu_id']) == 'mua']}
        ok = False
        for t in t_grid:
            for g, rs in groups.items():
                v = [r.get('clean_frac_when_present') for r in rs]
                v = [x for x in v if x is not None]
                if v:
                    out['stability'][t][g].append(100 * np.mean(np.asarray(v) >= t))
                    ok = True
        out['n_sessions_stability'] += ok

    todo = files if n_lratio_sessions is None else files[:n_lratio_sessions]
    for p in todo:
        j = json.loads(p.read_text())
        lab = ks_labels(j['ks_dir'])
        per = {}
        for h5p in sorted((p.parent / 'isolation').glob('*.SortData.h5')):
            try:
                clu = int(h5p.name.split('.')[0])
                with h5py.File(h5p, 'r') as g:
                    rows_ = []
                    for nm in sorted(k for k in g if k.startswith('frame_')):
                        fr = g[nm]
                        u = np.asarray(fr['unit']) if 'unit' in fr else None
                        nz = np.asarray(fr['noise']) if 'noise' in fr else None
                        if u is None or nz is None or not u.size or not nz.size:
                            rows_.append(np.nan)
                            continue
                        vals = [_compute_l_ratio(u[:, :3], nz[:, :3])]
                        oth = fr.get('other')
                        if oth is not None:
                            items = ([oth[k] for k in oth] if hasattr(oth, 'keys')
                                     else list(np.asarray(oth)))
                            for o in items:
                                o = np.asarray(o)
                                if o.ndim == 2 and o.shape[0]:
                                    vals.append(_compute_l_ratio(u[:, :3], o[:, :3]))
                        rows_.append(max(vals))
                    lr = np.asarray(rows_, float)
            except Exception:
                continue
            m = ~np.isnan(lr)
            if m.any():
                per[clu] = lr[m]
        if not per:
            continue
        out['n_sessions_lratio'] += 1
        for gl in l_grid:
            for g, sel in (('all', lambda c: True),
                           ('good', lambda c: lab.get(c) == 'good'),
                           ('mua', lambda c: lab.get(c) == 'mua')):
                v = [np.mean(lr < gl) >= SHIPPED_T_RATIO
                     for c, lr in per.items() if sel(c)]
                if v:
                    out['lratio'][gl][g].append(100 * float(np.mean(v)))
    return out


def make_pass_rate_panel(pr, out_svg):
    """Panel D: old manuscript Figure 5C/5D -- how many clusters clear each
    threshold, split by the sorter's own label.

    ONE COLOUR SET FOR BOTH HALVES. The original used different colours for
    All/Good/MUA in 5C and 5D, so the same group changed colour between two
    adjacent panels.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as _np
    from matplotlib.patches import Patch
    from . import style
    style.apply()
    SZ = style.SZ
    COL = {'all': '#C2185B', 'good': '#4FC3F7', 'mua': '#FBC02D'}
    NAME = {'all': 'All', 'good': 'KS4 good', 'mua': 'KS4 mua'}

    fig, (a, b) = plt.subplots(1, 2, figsize=(style.W_IN * 0.62, 2.05),
                               gridspec_kw=dict(wspace=0.34, left=0.095,
                                                right=0.99, top=0.84,
                                                bottom=0.22))
    rng = _np.random.default_rng(0)

    def draw(ax, table, keys, fmt, xlabel, title, shipped):
        for i, k in enumerate(keys):
            for j, g in enumerate(('all', 'good', 'mua')):
                v = _np.asarray(table[k][g], float)
                v = v[_np.isfinite(v)]
                if not v.size:
                    continue
                x = i + (j - 1) * 0.26
                bp = ax.boxplot([v], positions=[x], widths=0.18,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='k', lw=0.7),
                                whiskerprops=dict(lw=0.5),
                                capprops=dict(lw=0.5))
                for box in bp['boxes']:
                    box.set(facecolor='white', edgecolor='k', linewidth=0.5)
                ax.scatter(x + rng.uniform(-0.065, 0.065, v.size), v, s=1.6,
                           color=COL[g], alpha=0.7, linewidths=0, zorder=3)
            if k == shipped:
                ax.axvspan(i - 0.42, i + 0.42, color='#F2F4F7', zorder=0)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([fmt(k) for k in keys], fontsize=SZ['tick'])
        ax.set_xlim(-0.5, len(keys) - 0.5); ax.set_ylim(-4, 104)
        ax.set_xlabel(xlabel); ax.set_ylabel('Percentage of clusters (%)')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.6)
        ax.set_title(title, loc='left', fontsize=SZ['sub'])

    draw(a, pr['lratio'], pr['l_grid'], lambda k: f'≤ {k}',
         'L-ratio threshold', 'Isolation by L-ratio',
         pr['shipped']['l_ratio'])
    draw(b, pr['stability'], pr['t_grid'], lambda k: f'≥ {k:.0%}',
         'Stability level', 'Isolation stability',
         pr['shipped']['t_ratio'])
    a.legend(handles=[Patch(facecolor=COL[g], edgecolor='k', linewidth=0.5,
                            label=NAME[g]) for g in ('all', 'good', 'mua')],
             loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=SZ['tick'],
             frameon=True, handlelength=1.1, borderpad=0.3, labelspacing=0.25)
    a.text(0.98, 0.04, f"{pr['n_sessions_lratio']} sessions",
           transform=a.transAxes, ha='right', va='bottom',
           fontsize=SZ['tick'], color=style.C['neutral'])
    b.text(0.98, 0.04, f"{pr['n_sessions_stability']} sessions",
           transform=b.transAxes, ha='right', va='bottom',
           fontsize=SZ['tick'], color=style.C['neutral'])
    _save(fig, out_svg)
    return out_svg
