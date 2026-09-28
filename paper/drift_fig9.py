"""Figure 9, single-bank half: what the drift traces look like, and the corpus.

A  the lowest-drift single-bank session, all 9 blocks
B  the highest, once artifact-flagged sessions are excluded
C  range_90 against recording duration, every session
D  the same with the flagged sessions removed, and the fit

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


def make_single_bank_figure(low, high, rows, out_path):
    """low/high: dicts with name + ks_dir. rows: the drift_v2_sb session list."""
    from scipy import stats
    from unitassay.drift.metrics import load_dshift, range_90
    plt = _style()

    fig = plt.figure(figsize=(15.5, 10.4))
    gs = fig.add_gridspec(2, 2, hspace=.34, wspace=.22,
                          left=.065, right=.985, top=.915, bottom=.070)

    # ---- A and B: the traces themselves ------------------------------------
    d_lo = load_dshift(low['ks_dir'])
    d_hi = load_dshift(high['ks_dir'])
    # ONE offset for both panels, so the two are read on the same scale. Sized
    # on the HIGH session: scaled to the low one, its traces would overlap.
    hi_c = d_hi - np.median(d_hi, axis=0, keepdims=True)
    off = 2.4 * float(np.percentile(np.abs(hi_c), 98))

    for k, (spec, d, col, lab) in enumerate((
            (low, d_lo, LOW, 'A'), (high, d_hi, HIGH, 'B'))):
        ax = fig.add_subplot(gs[0, k])
        c = d - np.median(d, axis=0, keepdims=True)
        t = np.arange(d.shape[0]) * 2.0 / 60.0
        for b in range(d.shape[1]):
            ax.plot(t, c[:, b] + b * off, lw=.9, color=col, alpha=.85)
            ax.text(t[-1] * 1.012, b * off, str(b), fontsize=FS_LEG - 1,
                    va='center', ha='left', color=MUTED)
        ax.set_xlim(t[0] - t[-1] * .012, t[-1] * 1.16)
        ax.set_yticks([0, (d.shape[1] - 1) * off])
        ax.set_yticklabels(['block 0', f'block {d.shape[1] - 1}'])
        ax.set_xlabel('Time (min)', fontsize=FS_LAB)
        ax.set_ylabel('Drift, offset by block (µm)', fontsize=FS_LAB)
        # a scale bar, since the y axis is offset rather than metric
        x0 = t[-1] * 1.055
        ax.plot([x0, x0], [0, ROW_PITCH_UM], color=INK, lw=2.4,
                solid_capstyle='butt', clip_on=False)
        ax.text(x0 * 1.008, ROW_PITCH_UM / 2, '20 µm\n(one row)',
                fontsize=FS_LEG - 1, va='center', ha='left', color=INK,
                linespacing=1.25)
        r = float(range_90(d))
        ax.set_title(f'{lab}.  {"lowest" if lab == "A" else "highest"} drift'
                     f'  —  range$_{{90}}$ {r:.1f} µm',
                     loc='left', fontsize=FS_TITLE, weight='bold', color=INK,
                     pad=24)
        # under the title, not inside the axes -- in the axes it landed on
        # block 8's trace, which is where the top of the stack always is
        ax.text(0, 1.012, f"{spec['name']}   ·   {t[-1]:.1f} min, "
                          f"{d.shape[1]} blocks, single bank",
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=FS_ANN, color=MUTED)
        for s in ax.spines.values():
            s.set_color(GREY)

    # ---- C and D: the corpus, as in Fig_Drift_vs_RecDuration_SB ------------
    r90 = np.array([r['mean_r90'] for r in rows], float)
    dur = np.array([r['dur_min'] for r in rows], float)
    clean = np.array([r['clean'] for r in rows], bool)
    nc, ne = int(clean.sum()), int((~clean).sum())

    axc = fig.add_subplot(gs[1, 0])
    axc.scatter(dur[~clean], r90[~clean], s=70, marker='^', color=GREY,
                edgecolor='white', lw=.8, label=f'artifact-flagged ({ne})', zorder=2)
    axc.scatter(dur[clean], r90[clean], s=52, color=LOW, alpha=.75,
                edgecolor='white', lw=.5, label=f'clean ({nc})', zorder=3)
    rr, pp = stats.pearsonr(dur, r90)
    axc.text(.97, .955, f'N={len(rows)}\nr={rr:+.2f}  p={pp:.2f}',
             transform=axc.transAxes, ha='right', va='top', fontsize=FS_ANN,
             color=MUTED)
    axc.set_yscale('log')
    axc.set_xlabel('Recording duration (min)', fontsize=FS_LAB)
    axc.set_ylabel('range$_{90}$ (µm)', fontsize=FS_LAB)
    axc.set_title('C.  all single-bank sessions', loc='left',
                  fontsize=FS_TITLE, weight='bold', color=INK)
    axc.legend(fontsize=FS_LEG, frameon=False, loc='lower right')
    for s in axc.spines.values():
        s.set_color(GREY)

    axd = fig.add_subplot(gs[1, 1])
    axd.scatter(dur[clean], r90[clean], s=58, color=LOW, alpha=.75,
                edgecolor='white', lw=.5, zorder=3)
    sl, ic = np.polyfit(dur[clean], r90[clean], 1)
    xs = np.linspace(dur[clean].min(), dur[clean].max(), 20)
    axd.plot(xs, sl * xs + ic, '--', color='#c2410c', lw=2.2, zorder=4)
    rc, pc = stats.pearsonr(dur[clean], r90[clean])
    axd.text(.97, .97, f'N={nc}\nr={rc:+.2f}  p={pc:.3f}\nslope={sl:.2f} µm/min',
             transform=axd.transAxes, ha='right', va='top', fontsize=FS_ANN,
             color=MUTED)
    axd.axhline(ROW_PITCH_UM, color='#0AAA4B', lw=1.4, ls='--')
    axd.text(.015, ROW_PITCH_UM * 0.975, 'one 20 µm row',
             transform=axd.get_yaxis_transform(), ha='left', va='top',
             fontsize=FS_LEG, color='#0AAA4B')
    axd.set_xlabel('Recording duration (min)', fontsize=FS_LAB)
    axd.set_ylabel('range$_{90}$ (µm)', fontsize=FS_LAB)
    axd.set_title('D.  clean sessions only', loc='left', fontsize=FS_TITLE,
                  weight='bold', color=INK)
    for s in axd.spines.values():
        s.set_color(GREY)

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
