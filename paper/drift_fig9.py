"""Figure 9, single-bank half: what the drift traces look like, and the corpus.

A  a quiet single-bank recording, all 9 blocks
B  a single-bank recording of similar length with more drift
C  range_90 against recording duration, clean sessions, with the fit

ONE SCATTER, NOT TWO. The all-sessions panel and the clean-sessions panel
showed the same 68 points twice over, differing only by seven artifact
sessions that the incident supplementary already accounts for. C keeps the
clean set, which is the one the regression is run on.

CONCATENATED SESSIONS ARE DEFERRED. 89 single-bank concatenations exist, up
to 141 minutes, but all on the plain _KS4 arm -- putting one beside nb5
panels would mix arms. The CatGT versions are specified in
docs/catgt_concatenation_tasks.md.

REAL DATA ONLY (Agrita, 2026-09-28). The synthetic calibration validates KS4's
estimator, which is not the contribution being made here; it moves to
supplementary and is cited in one line, because the floor it measures -- 0.58
um single-bank -- is what tells a reader that 6 um is a measurement rather than
the estimator talking to itself.

Type and colour follow Fig_Drift_vs_RecDuration_SB_nb5 exactly, so this figure
and the drift_v2 originals can sit side by side: Arial, axis labels 13, panel
titles 14.5 bold, annotations 12.5, legend 11.5.
"""
import numpy as np

INK, MUTED, GREY = '#0D1B4B', '#546E7A', '#9aa7b4'

# BLOCK RAMP: red -> blue, shallow to deep (Agrita). Nine lines of one hue are
# hard to tell apart where they cross; two opposed hues separate the ends at a
# glance. It runs through PURPLE rather than through white: RdBu and coolwarm
# both pass near-white at their midpoint, which would make the middle block
# invisible against the page. The trade is that a red-blue ramp is not
# lightness-monotonic, so it reads as ordered rather than as a magnitude --
# which is right here, since block index is a position on the probe.
BLOCK_RAMP = ('#C2185B', '#9B2A8F', '#6B3BA8', '#3953A4')
LOW, HIGH = '#3953A4', '#C2185B'
FS_LAB, FS_TITLE, FS_ANN, FS_LEG, FS_TICK = 13, 14.5, 12.5, 11.5, 11.5
ROW_PITCH_UM = 20.0


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'axes.linewidth': 0.9})
    return plt


def _traces(ax, ks_dir, title, colour, batch_s=2.0):
    """One dshift trace per block, offset so nine can be told apart."""
    from unitassay.drift.metrics import load_dshift, range_90, _prctile
    d = load_dshift(ks_dir)
    nb = d.shape[1]
    t = np.arange(d.shape[0]) * batch_s / 60.0
    centred = d - np.median(d, axis=0, keepdims=True)
    # a common offset across BOTH panels is set by the caller; here we only
    # need it large enough that the traces of this session do not collide
    return d, nb, t, centred


def draw_single_bank_row(fig, axes, low, high, rows):
    """The three single-bank panels, into three axes someone else owns.

    Split out of make_single_bank_figure so Figure 9 can put this row above the
    multi-bank mechanism row and still be ONE figure.
    """
    from scipy import stats
    from unitassay.drift.metrics import load_dshift, range_90
    plt = _style()
    _ax = iter(axes)

    # ---- A and B: the traces themselves ------------------------------------
    d_lo = load_dshift(low['ks_dir'])
    d_hi = load_dshift(high['ks_dir'])
    # ONE offset for both panels, so the two are read on the same scale. Sized
    # on the HIGH session: scaled to the low one, its traces would overlap.
    hi_c = d_hi - np.median(d_hi, axis=0, keepdims=True)
    off = 2.4 * float(np.percentile(np.abs(hi_c), 98))

    # OVERLAID, one line per block, in the idiom of Fig_KS4_Drift_Algorithm
    # panel 6: a Blues ramp shallow -> deep, each line numbered at its right
    # end. Both panels share ONE y range, set by the high session -- scaled
    # independently they would look alike, which is the opposite of the point.
    import matplotlib.pyplot as _plt
    ylim = float(np.percentile(np.abs(hi_c), 99.5)) * 1.25
    for k, (spec, d, lab) in enumerate(((low, d_lo, 'A'), (high, d_hi, 'B'))):
        ax = next(_ax)
        c = d - np.median(d, axis=0, keepdims=True)
        t = np.arange(d.shape[0]) * 2.0 / 60.0
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('blockramp', BLOCK_RAMP)
        ramp = cmap(np.linspace(0, 1, d.shape[1]))
        # NO PER-LINE NUMBERS. Panel 6 of the KS4-algorithm figure can label
        # each line because its blocks separate in dshift; on a single-bank
        # session they end within a micron of each other and nine numbers pile
        # into one blob. The ramp carries depth instead, with a colourbar.
        for b in range(d.shape[1]):
            ax.plot(t, c[:, b], '-', color=ramp[b], lw=1.2, zorder=3)
        ax.axhspan(-ROW_PITCH_UM / 2, ROW_PITCH_UM / 2, color='#0AAA4B',
                   alpha=.08, lw=0, zorder=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=.25)
        ax.set_xlabel('Time (min)', fontsize=FS_LAB)
        ax.set_ylabel('dshift (µm)', fontsize=FS_LAB)
        r = float(range_90(d))
        ax.set_title(f"{lab}.  {spec['what']}  —  range$_{{90}}$ {r:.1f} µm",
                     loc='left', fontsize=FS_TITLE, weight='bold', color=INK,
                     pad=24)
        ax.text(0, 1.012, f"{spec['name']}   ·   {t[-1]:.1f} min, "
                          f"{d.shape[1]} blocks, {spec['geom']}",
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=FS_ANN, color=MUTED)
        # A LEGEND, not just a caption: three sampled blocks name the ends of
        # the ramp and its middle, so a reader can place any line without
        # counting colours.
        from matplotlib.lines import Line2D
        nbk = d.shape[1]
        mid = nbk // 2
        handles = [Line2D([], [], color=ramp[0], lw=2.0,
                          label=f'block 1 — shallow'),
                   Line2D([], [], color=ramp[mid], lw=2.0,
                          label=f'block {mid + 1}'),
                   Line2D([], [], color=ramp[-1], lw=2.0,
                          label=f'block {nbk} — deep'),
                   Line2D([], [], color='#0AAA4B', lw=6, alpha=.25,
                          label='one 20 µm row')]
        ax.legend(handles=handles, fontsize=FS_LEG, frameon=False,
                  loc='lower left', ncol=2, handlelength=1.5,
                  columnspacing=1.2, labelspacing=.35)
        # NO COLOURBAR. One per panel crowded both and its label sat on B's
        # y axis; a single shared one still costs width that the traces want.
        # The ramp is self-evident once the caption names its direction.
        for sp_ in ax.spines.values():
            sp_.set_color(GREY)

    # ---- C: the corpus, clean sessions, with the fit -----------------------
    r90 = np.array([r['mean_r90'] for r in rows], float)
    dur = np.array([r['dur_min'] for r in rows], float)
    clean = np.array([r['clean'] for r in rows], bool)
    nc, ne = int(clean.sum()), int((~clean).sum())

    axc = next(_ax)
    # BOTH SETS SHOWN, FIT ON CLEAN ONLY (Agrita). Hiding the artifact
    # sessions makes the corpus look tidier than it is; including them in the
    # fit would let a handful of mechanical incidents set the slope. They are
    # drawn as open triangles so a reader can see how few there are and how
    # far out they sit -- the incident supplementary accounts for them.
    axc.scatter(dur[~clean], r90[~clean], s=72, marker='^', facecolor='none',
                edgecolor=GREY, lw=1.2, zorder=2,
                label=f'artifact-flagged ({ne}) — excluded from the fit')
    axc.scatter(dur[clean], r90[clean], s=64, color=LOW, alpha=.75,
                edgecolor='white', lw=.6, zorder=3,
                label=f'clean ({nc}) — fitted')
    sl, ic = np.polyfit(dur[clean], r90[clean], 1)
    xs = np.linspace(dur[clean].min(), dur[clean].max(), 20)
    axc.plot(xs, sl * xs + ic, '--', color='#c2410c', lw=2.4, zorder=4)
    rc, pc = stats.pearsonr(dur[clean], r90[clean])
    # upper right: the legend holds the lower left and the artifact triangles
    # hold the upper left, so this is the only corner left
    axc.text(.985, .96, f'fit on the {nc} clean sessions\n'
                        f'r={rc:+.2f}   p={pc:.3f}   slope={sl:.2f} µm/min',
             transform=axc.transAxes, ha='right', va='top', fontsize=FS_ANN,
             color=MUTED, linespacing=1.35)
    axc.margins(y=.14)
    axc.axhline(ROW_PITCH_UM, color='#0AAA4B', lw=1.6, ls='--')
    axc.text(.012, ROW_PITCH_UM * 0.955, 'one 20 µm row',
             transform=axc.get_yaxis_transform(), ha='left', va='top',
             fontsize=FS_LEG, color='#0AAA4B')
    # log y: the artifact sessions reach 117 um and would flatten everything
    # else onto the axis
    axc.set_yscale('log')
    axc.legend(fontsize=FS_LEG, frameon=False, loc='lower left')
    axc.set_xlabel('Recording duration (min)', fontsize=FS_LAB)
    axc.set_ylabel('range$_{90}$ (µm)', fontsize=FS_LAB)
    axc.set_title('C.  no growth detected with recording length',
                  loc='left', fontsize=FS_TITLE, weight='bold', color=INK,
                  pad=10)
    for sp_ in axc.spines.values():
        sp_.set_color(GREY)

    return dict(fit_r=float(rc), fit_p=float(pc), fit_slope=float(sl),
                n_clean=nc, n_flagged=ne)


def make_single_bank_figure(low, high, rows, out_path):
    """The single-bank row on its own, for checking it in isolation."""
    plt = _style()
    fig = plt.figure(figsize=(15.5, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=.36, wspace=.20,
                          height_ratios=[1.0, .92],
                          left=.065, right=.985, top=.915, bottom=.072)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]
    draw_single_bank_row(fig, axes, low, high, rows)
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


# ---------------------------------------------------------------------------
# Figure 9 as one row of examples plus the population, all single bank.
#
# SINGLE BANK THROUGHOUT, and that is what lets the multi-bank mechanism go to
# supplementary cleanly: a single-bank probe has no inter-bank gap, so bug 5
# cannot arise in any panel here. No caveat crosses the boundary.
#
# The exemplars are MARKED IN THE SCATTER, so a reader can see where each one
# sits in the population instead of taking the choice on trust.

EX_COLOURS = {'quiet': '#0A8F40', 'drift': '#3953A4',
              'step': '#C2185B', 'coherent': '#E67E22'}


def draw_trace_panel(ax, ks_dir, title, subtitle, ylim=None, batch_s=2.0,
                     note=None):
    """One session's dshift, all blocks overlaid. Returns (range_90, pairwise r).

    The pairwise correlation is reported because it is the finding the overlap
    hides: on a quiet session the nine blocks are UNCORRELATED (independent
    estimator noise, nothing to separate), on a drifting one they are almost
    perfectly correlated (the probe moves as a rigid body). Either way the
    traces coincide, and without r a reader just sees 'I cannot count nine'.
    """
    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap
    from unitassay.drift.metrics import load_dshift, range_90
    d = load_dshift(ks_dir)
    c = d - _np.median(d, axis=0, keepdims=True)
    t = _np.arange(d.shape[0]) * batch_s / 60.0
    nb = d.shape[1]
    # ONE SCHEME ON EVERY DSHIFT PANEL (Agrita). The artifact panels used to
    # carry their own accent colour, which made them look like a different
    # measurement rather than the same one behaving badly.
    cmap = LinearSegmentedColormap.from_list('blockramp', BLOCK_RAMP)
    cols = cmap(_np.linspace(0, 1, nb))
    for b in range(nb):
        ax.plot(t, c[:, b], '-', color=cols[b], lw=1.0, zorder=3)
    ax.axhspan(-ROW_PITCH_UM / 2, ROW_PITCH_UM / 2, color='#0AAA4B', alpha=.08,
               lw=0, zorder=1)
    ax.set_xlim(t[0], t[-1])
    if ylim:
        ax.set_ylim(-ylim, ylim)
    ax.grid(alpha=.25)
    r = float(range_90(d))
    rr = float(_np.median([_np.corrcoef(c[:, i], c[:, j])[0, 1]
                           for i in range(nb) for j in range(i + 1, nb)]))
    ax.set_title(title, loc='left', fontsize=FS_TITLE, weight='bold',
                 color=INK, pad=22)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FS_ANN, color=MUTED)
    txt = f'range$_{{90}}$ {r:.1f} µm\nblocks r = {rr:+.2f}'
    if note:
        txt += f'\n{note}'
    ax.text(.985, .04, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FS_ANN, color=MUTED, linespacing=1.3)
    for sp_ in ax.spines.values():
        sp_.set_color(GREY)
    return r, rr


def draw_population_panel(ax, rows, highlights):
    """range_90 against duration, with the example sessions marked.

    `highlights` maps session name -> (colour, label). Marking them is the
    point: it shows where each example sits in the population, so a reader can
    see that the quiet one is typical and the artifacts are the tail, rather
    than taking the choice of exemplar on trust.
    """
    import numpy as _np
    from scipy import stats
    r90 = _np.array([r['mean_r90'] for r in rows], float)
    dur = _np.array([r['dur_min'] for r in rows], float)
    clean = _np.array([r['clean'] for r in rows], bool)
    name = [r['name'] for r in rows]
    nc, ne = int(clean.sum()), int((~clean).sum())

    ax.scatter(dur[~clean], r90[~clean], s=64, marker='^', facecolor='none',
               edgecolor=GREY, lw=1.1, zorder=2,
               label=f'artifact-flagged ({ne})')
    ax.scatter(dur[clean], r90[clean], s=52, color='#9fb0c0', alpha=.85,
               edgecolor='white', lw=.5, zorder=3, label=f'clean ({nc})')
    sl, ic = _np.polyfit(dur[clean], r90[clean], 1)
    xs = _np.linspace(dur[clean].min(), dur[clean].max(), 20)
    ax.plot(xs, sl * xs + ic, '--', color='#c2410c', lw=2.2, zorder=4)
    rc, pc = stats.pearsonr(dur[clean], r90[clean])

    # A GROUP GETS ONE ENTRY AND ALL ITS MARKERS. The 2020-08-01 event is four
    # probes recording at once; starring only one of them showed a single
    # outlier where the point is that FOUR moved together.
    for lab, (col, members) in highlights.items():
        first = True
        for nm in members:
            if nm not in name:
                continue
            i = name.index(nm)
            ax.scatter([dur[i]], [r90[i]], s=170, marker='*', color=col,
                       edgecolor='white', lw=.9, zorder=6,
                       label=lab if first else None)
            first = False
    ax.set_yscale('log')
    ax.axhline(ROW_PITCH_UM, color='#0AAA4B', lw=1.4, ls='--', zorder=1)
    ax.text(.012, ROW_PITCH_UM * .95, 'one 20 µm row',
            transform=ax.get_yaxis_transform(), ha='left', va='top',
            fontsize=FS_LEG, color='#0AAA4B')
    ax.text(.985, .96, f'fit on the {nc} clean sessions\n'
                       f'r={rc:+.2f}  p={pc:.3f}  slope={sl:.2f} µm/min',
            transform=ax.transAxes, ha='right', va='top', fontsize=FS_ANN,
            color=MUTED, linespacing=1.3)
    ax.legend(fontsize=FS_LEG - .5, frameon=False, loc='lower left', ncol=2,
              handletextpad=.4, columnspacing=1.0)
    ax.margins(y=.16)
    ax.set_xlabel('Recording duration (min)', fontsize=FS_LAB)
    ax.set_ylabel('range$_{90}$ (µm)', fontsize=FS_LAB)
    ax.set_title('D.  the population', loc='left', fontsize=FS_TITLE,
                 weight='bold', color=INK, pad=10)
    for sp_ in ax.spines.values():
        sp_.set_color(GREY)
    return dict(r=float(rc), p=float(pc), slope=float(sl), n_clean=nc, n_flag=ne)


def draw_block_legend(ax, nb=9):
    """The ramp as nine entries, one per block — Agrita asked for all nine.

    Three sampled blocks left a reader interpolating; nine shows the ordering
    is the whole scale and that no block is special.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    import numpy as _np
    cmap = LinearSegmentedColormap.from_list('blockramp', BLOCK_RAMP)
    cols = cmap(_np.linspace(0, 1, nb))
    handles = [Line2D([], [], color=cols[i], lw=2.6,
                      label=f'{i + 1}' + (' shallow' if i == 0 else
                                          ' deep' if i == nb - 1 else ''))
               for i in range(nb)]
    handles.append(Line2D([], [], color='#0AAA4B', lw=7, alpha=.25,
                          label='one 20 µm row'))
    ax.axis('off')
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, ncol=5,
              loc='center', handlelength=1.6, columnspacing=1.1,
              labelspacing=.5, title='drift block, shallow → deep',
              title_fontsize=FS_LEG)
