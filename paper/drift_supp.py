"""Drift supplementary: the single-bank corpus, and the incidents inside it.

TWO FIGURES, ONE ARGUMENT. The corpus panel shows that drift on a single-bank
probe is small and does not grow with recording length. The incident panel
shows that the handful of sessions above the outlier fence are not evidence
against that: they are identifiable mechanical events, and the same probes on
the same day recover within hours.

WHY THE INCIDENTS ARE LAID OUT AS RECORDING x PROBE. Drawn as seven
independent panels the point is invisible. Arranged so that simultaneous
probes sit in a row, it reads itself: on 2020-08-01 all four probes drift
together in recording 001 -- which no probe fault can do -- one probe is
affected in 002, and by 003 the same four read 1.6 to 2.6 um. On 2022-05-23
the signature is the opposite and just as legible: np1 alone is elevated in
all three recordings while np2 and np3 stay normal, i.e. one probe seated
badly for the day rather than a transient event.
"""
import numpy as np

C_OK = '#6175B6'
C_HOT = '#C2185B'
C_FENCE = '#BF1E2D'
C_ROW = '#0A8F40'
INK = '#111418'
GREY = '#5B6470'
ROW_PITCH_UM = 20.0        # one Neuropixels row: the scale bar for "does this matter"


def tukey_fence(values):
    """Upper Tukey fence, q3 + 1.5 IQR -- the cut drawn on the corpus panel.

    Stated rather than chosen by eye: the outlier set should be reproducible
    from the numbers, not from where a reader thinks the gap is.
    """
    v = np.asarray(values, float)
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def make_corpus_panel(sessions, out_path, metric='r90'):
    """Single-bank drift against recording duration, with the fence marked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    sb = [s for s in sessions if s['cls'] == 0]
    x = np.array([s['dur'] for s in sb], float)
    y = np.array([s[metric] for s in sb], float)
    fence = tukey_fence(y)
    hot = y > fence

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.subplots_adjust(left=.115, right=.975, top=.90, bottom=.135)
    ax.axhspan(0, ROW_PITCH_UM, color=C_ROW, alpha=.07, lw=0)
    ax.axhline(ROW_PITCH_UM, color=C_ROW, lw=1.0, ls='--')
    ax.axhline(fence, color=C_FENCE, lw=1.0, ls=':')
    ax.scatter(x[~hot], y[~hot], s=26, color=C_OK, alpha=.85, linewidths=0,
               label=f'within the fence  (n={int((~hot).sum())})')
    ax.scatter(x[hot], y[hot], s=34, color=C_HOT, alpha=.95, linewidths=0,
               label=f'above the fence  (n={int(hot.sum())})')
    # LABELS STACKED BY X-GROUP. Sessions recorded simultaneously share a
    # duration exactly -- the four probes of 200801_001 are one x value and
    # two of them sit 6% apart in log y -- so alternating a fixed offset still
    # overlapped them. Labels are laddered within each group of equal x.
    import collections as _c
    groups = _c.defaultdict(list)
    for s_, xx, yy, h in zip(sb, x, y, hot):
        if h:
            groups[round(float(xx), 1)].append((yy, s_['name'], xx))
    for _gx, items in groups.items():
        items.sort(key=lambda t: -t[0])
        for k, (yy, nm, xx) in enumerate(items):
            ax.annotate(nm, (xx, yy), textcoords='offset points',
                        xytext=(11, 6 - 11 * k), fontsize=6.0, color=C_HOT,
                        va='center', ha='left',
                        arrowprops=dict(arrowstyle='-', lw=.5, color=C_HOT,
                                        alpha=.55, shrinkA=0, shrinkB=2))

    ax.set_yscale('log')
    ax.set_xlabel('recording duration (min)', fontsize=9, color=INK)
    ax.set_ylabel('drift, range$_{90}$ (µm)', fontsize=9, color=INK)
    # the two reference lines sit 2.6 um apart, so their labels go on
    # opposite sides of the axis rather than on top of one another
    ax.text(.985, ROW_PITCH_UM * 1.10, 'one 20 µm row',
            transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            fontsize=7, color=C_ROW)
    ax.text(.015, fence * 0.88, f'outlier fence  {fence:.1f} µm',
            transform=ax.get_yaxis_transform(), ha='left', va='top',
            fontsize=7, color=C_FENCE)
    med = float(np.median(y))
    ax.set_title(f'Single-bank sessions (n={len(sb)}).  Median {med:.1f} µm; '
                 f'{100 * np.mean(y < ROW_PITCH_UM):.0f}% below one row pitch.',
                 fontsize=9.5, color=INK, loc='left', pad=8)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    ax.tick_params(labelsize=8, colors=INK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color('#C7CCD4')
    return _save(fig, out_path)


def make_incident_panel(traces, out_path, title='', flags=None):
    """dshift traces laid out recording x probe.

    `traces` is {(recording, probe): (array (n_batches, n_blocks), r90)};
    `flags` the artifact record per (recording, probe).

    THE FLAGS ARE THE POINT, not decoration. These traces are not slow drift:
    on 2020-08-01 between 76% and 100% of blocks step TOGETHER, which tissue
    cannot do, and the C1 fast-coherent criterion -- defined independently of
    this figure -- already flags every one of them. Labelling each panel with
    that classification turns "these sessions are outliers" into "these
    sessions are identified artifacts", which is a much stronger statement and
    one the reader can check.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    recs = sorted({k[0] for k in traces})
    probes = sorted({k[1] for k in traces}, key=int)
    fig, axes = plt.subplots(len(recs), len(probes),
                             figsize=(2.55 * len(probes), 1.95 * len(recs)),
                             squeeze=False, sharex=False)
    fig.subplots_adjust(left=.085, right=.985, top=.815, bottom=.095,
                        wspace=.28, hspace=.42)
    ymax = max(np.nanmax(np.abs(t[0] - np.nanmedian(t[0]))) for t in traces.values())
    for i, rec in enumerate(recs):
        for j, pr in enumerate(probes):
            ax = axes[i][j]
            item = traces.get((rec, pr))
            if item is None:
                ax.axis('off')
                continue
            d, r90 = item
            t = np.arange(d.shape[0]) * 2.0 / 60.0          # 2 s batches -> min
            centred = d - np.nanmedian(d, axis=0, keepdims=True)
            hot = r90 > 17.4
            for b in range(d.shape[1]):
                ax.plot(t, centred[:, b], lw=0.7,
                        color=(C_HOT if hot else C_OK), alpha=.75)
            ax.set_ylim(-ymax * 1.05, ymax * 1.05)
            ax.axhspan(-ROW_PITCH_UM / 2, ROW_PITCH_UM / 2, color=C_ROW,
                       alpha=.09, lw=0)
            ax.set_title(f'np{pr}   range$_{{90}}$ {r90:.1f} µm', fontsize=8,
                         color=(C_HOT if hot else INK), pad=3)
            fl = (flags or {}).get((rec, pr))
            if fl:
                if fl.get('is_fast_coherent'):
                    note = (f"C1 fast-coherent artifact\n"
                            f"{100 * fl.get('coh_frac', 0):.0f}% of blocks step together")
                    col = C_HOT
                elif fl.get('is_step_disp'):
                    note = 'step displacement'
                    col = C_HOT
                else:
                    note = 'no artifact flag'
                    col = GREY
                ax.text(.02, .03, note, transform=ax.transAxes, fontsize=6.0,
                        va='bottom', ha='left', color=col, linespacing=1.25)
            if j == 0:
                ax.set_ylabel(f'rec {rec}\ndrift (µm)', fontsize=8, color=INK)
            if i == len(recs) - 1:
                ax.set_xlabel('time (min)', fontsize=8, color=INK)
            ax.tick_params(labelsize=6.5, colors=INK)
            for sp in ax.spines.values():
                sp.set_linewidth(0.7); sp.set_color('#C7CCD4')
    # wrapped, because the one-line version ran off the right edge
    fig.suptitle(title, fontsize=9.5, x=.085, ha='left', y=.985, color=INK,
                 linespacing=1.35)
    return _save(fig, out_path)


def _save(fig, out_path):
    import matplotlib.pyplot as plt
    out_path = str(out_path)
    fig.savefig(out_path, facecolor='white')
    for ext in ('.png', '.pdf'):
        fig.savefig(out_path.rsplit('.', 1)[0] + ext, facecolor='white',
                    dpi=200 if ext == '.png' else None)
    plt.close(fig)
    if out_path.endswith('.svg'):
        from .svg_expand import expand_use, report
        expand_use(out_path)
        return report(out_path)
