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
    n_clu = 0
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
            for g_ in l_grid:
                passed[g_] += int(np.mean(lr[ok] < g_) >= SHIPPED_T_RATIO)

    return dict(
        shipped=dict(l_ratio=SHIPPED_L_RATIO, t_ratio=SHIPPED_T_RATIO),
        stability=dict(t=t_grid.tolist(),
                       pct_pass=[float(100 * np.mean(frac >= t)) for t in t_grid],
                       n=int(frac.size)),
        lratio=dict(l=list(l_grid),
                    pct_pass=[100 * passed[g] / max(n_clu, 1) for g in l_grid],
                    n=n_clu, n_sessions=n_lratio_sessions))


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
def make_threshold_panel(sweep, out_svg):
    """Panel D: how the pass rate moves with each shipped threshold."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    C, SZ = style.C, style.SZ
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.W_IN * 0.52, 1.95),
                               gridspec_kw=dict(wspace=0.46, left=0.11,
                                                right=0.985, top=0.84,
                                                bottom=0.30))
    L = sweep['lratio']
    a.plot(L['l'], L['pct_pass'], '-o', ms=2.4, lw=1.0, color=C['teal'])
    a.axvline(sweep['shipped']['l_ratio'], color=C['warm'], lw=0.9,
              ls=(0, (3, 2)))
    a.set_xscale('log')
    # a tick at every swept value collides on a log axis; label the decades
    # plus the shipped point, and let the markers show where the rest sit
    ticks = [0.05, 0.1, sweep['shipped']['l_ratio'], 0.4, 1.0]
    a.set_xticks(ticks); a.set_xticklabels([str(t) for t in ticks])
    a.minorticks_off()
    a.set_xlabel('L-ratio threshold'); a.set_ylabel('% of clusters passing')
    a.set_ylim(0, 100)
    a.set_title(f"(i) L-ratio · shipped {sweep['shipped']['l_ratio']}",
                loc='left', fontsize=SZ['tick'])
    a.text(0.97, 0.06, f"n = {L['n']:,}", transform=a.transAxes, ha='right',
           fontsize=SZ['tick'], color=C['neutral'])

    S = sweep['stability']
    b.plot(S['t'], S['pct_pass'], '-', lw=1.1, color=C['teal'])
    b.axvline(sweep['shipped']['t_ratio'], color=C['warm'], lw=0.9,
              ls=(0, (3, 2)))
    b.set_xlabel('stability threshold\n(fraction of frames clean)',
                 linespacing=1.4)
    b.set_ylabel('% of clusters passing'); b.set_ylim(0, 100)
    b.set_title(f"(ii) stability · shipped {sweep['shipped']['t_ratio']}",
                loc='left', fontsize=SZ['tick'])
    b.text(0.97, 0.9, f"n = {S['n']:,}", transform=b.transAxes, ha='right',
           fontsize=SZ['tick'], color=C['neutral'])
    fig.savefig(out_svg, facecolor='white')
    plt.close(fig)
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
    fig.savefig(out_svg, facecolor='white')
    plt.close(fig)
    return out_svg
