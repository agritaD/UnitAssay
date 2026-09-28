"""Figure 9 (drift), main panels.

B  the calibration    what KS4's number means, measured against known truth
C  the prevalence     how much drift is actually in this corpus
D  the cost           what drift does to the over-merge test, and why it
                      cannot be corrected away

THREE THINGS THIS FIGURE MUST NOT SAY, two of which already caught me out:

* Panel C does NOT show that KS4 handles drift well. It shows there is little
  drift HERE -- a fact about the recordings, not about the sorter.
* KS4's CORRECTION works. The C2 oracle settled it: uncorrected, recovery
  collapses from 292 to 63 units at 80 um; corrected, it is flat and matches a
  perfect-estimate oracle exactly. Panel D's cost is to OUR over-merge test.
* P1's drift-free headline of 12.9% was an artefact of oversized clusters. At
  realistic cluster size it is 3.37%, and that is the number panel D uses.
"""
import numpy as np

C_UNIT = '#6175B6'
C_HOT = '#C2185B'
C_TRUE = '#111418'
C_ROW = '#0A8F40'
C_KS4 = '#7A828C'
INK = '#111418'
GREY = '#5B6470'
ROW_PITCH_UM = 20.0
KS4_HIGH_DRIFT_UM = 37.0     # the KS4 paper's own high-drift condition


def _finish(fig, out_path):
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


def _axes(ax):
    ax.tick_params(labelsize=7.5, colors=INK)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color('#C7CCD4')


def make_calibration_panel(floors, ladder, out_path):
    """B: the floor, the gap-block bias, and the estimate against known truth.

    The part a KS4 user cannot get anywhere else. `ops['dshift']` arrives with
    no scale: 6 um might be drift or might be the estimator talking to itself.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                 gridspec_kw=dict(width_ratios=[1, 1.25]))
    fig.subplots_adjust(left=.085, right=.985, top=.845, bottom=.165, wspace=.32)

    # --- B(i): what the estimator reports when the true answer is ZERO
    labels = ['single bank', 'multi-bank\npopulated', 'multi-bank\nGAP block']
    vals = [floors['single_bank'], floors['multi_populated'], floors['multi_gap']]
    cols = [C_UNIT, C_UNIT, C_HOT]
    a1.bar(range(3), vals, width=.62, color=cols, lw=0)
    for i, v in enumerate(vals):
        a1.text(i, v + .05, f'{v:.2f}', ha='center', va='bottom', fontsize=8.5,
                color=cols[i], weight='semibold')
    a1.axhline(0, color=C_TRUE, lw=1.2)
    a1.text(1.0, -0.115, 'all three: true drift = 0', ha='center', va='top',
            fontsize=7.5, color=C_TRUE, style='italic',
            transform=a1.get_xaxis_transform())
    ratio = floors['multi_gap'] / max(floors['multi_populated'], 1e-9)
    a1.annotate(f'{ratio:.1f}× the populated floor\n— bias, not measurement',
                xy=(2, floors['multi_gap']), xytext=(1.35, floors['multi_gap'] * 1.28),
                fontsize=7.4, color=C_HOT, ha='center',
                arrowprops=dict(arrowstyle='-|>', color=C_HOT, lw=.9))
    a1.set_xticks(range(3)); a1.set_xticklabels(labels, fontsize=7.5)
    a1.set_ylabel('reported range$_{90}$ (µm)', fontsize=8.5, color=INK)
    a1.set_ylim(0, floors['multi_gap'] * 1.55)
    a1.set_title('The floor: what is reported when there is nothing to report',
                 fontsize=8.8, loc='left', color=INK, pad=7)
    _axes(a1)

    # --- B(ii): the estimate against known truth
    t = np.array([r['true_range_90'] for r in ladder], float)
    e = np.array([r['est_range_90'] for r in ladder], float)
    lim = max(t.max(), e.max()) * 1.08
    a2.plot([0, lim], [0, lim], color=C_TRUE, lw=1.0, ls='--', zorder=1)
    a2.plot(t, e, '-', color=C_UNIT, lw=1.2, zorder=2)
    a2.scatter(t, e, s=42, color=C_UNIT, zorder=3, linewidths=0)
    for tt, ee in zip(t, e):
        if tt > 0:
            a2.annotate(f'{ee / tt:.2f}×', (tt, ee), textcoords='offset points',
                        xytext=(7, -9), fontsize=6.8, color=GREY)
    a2.axvspan(0, ROW_PITCH_UM, color=C_ROW, alpha=.07, lw=0)
    a2.axvline(ROW_PITCH_UM, color=C_ROW, lw=1.0, ls='--')
    a2.text(ROW_PITCH_UM, lim * .02, ' one 20 µm row', fontsize=7,
            color=C_ROW, va='bottom')
    a2.set_xlim(-1, lim); a2.set_ylim(-1, lim)
    a2.set_xlabel('true drift imposed (µm)', fontsize=8.5, color=INK)
    a2.set_ylabel('KS4 estimate, range$_{90}$ (µm)', fontsize=8.5, color=INK)
    a2.set_title('The estimate is unbiased across the whole range',
                 fontsize=8.8, loc='left', color=INK, pad=7)
    _axes(a2)
    return _finish(fig, out_path)


def make_prevalence_panel(sessions, out_path):
    """C: how much drift this corpus actually has, by probe geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    CLS = {0: 'single bank', 1: 'multi-bank\nno gap block', 2: 'multi-bank\n≥1 gap block'}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fig.subplots_adjust(left=.125, right=.975, top=.805, bottom=.135)

    # GAP BLOCKS ARE EXCLUDED. A block that lands in a multi-bank probe's dead
    # gap reports interpolation, not measurement -- 1.67 um where the true
    # answer is verified zero, as panel B shows -- so the session value must be
    # the mean over POPULATED blocks. Plotting r90 instead put the gap class at
    # 7.9 um against its true 6.9, and inflated the multi-vs-single contrast
    # from 1.50x to 1.66x. r90_pop is identical to r90 for the two classes that
    # have no gap block, so only the third bar moves.
    rng = np.random.default_rng(0)
    allv = np.array([s['r90_pop'] for s in sessions], float)
    for i, c in enumerate((0, 1, 2)):
        v = np.array([s['r90_pop'] for s in sessions if s['cls'] == c], float)
        x = i + rng.uniform(-.17, .17, v.size)
        ax.scatter(x, v, s=17, color=C_UNIT, alpha=.65, linewidths=0)
        bp = ax.boxplot([v], positions=[i], widths=.46, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.2),
                        whiskerprops=dict(lw=.7), capprops=dict(lw=.7))
        for b in bp['boxes']:
            b.set(facecolor='none', edgecolor='#8A8F98', lw=.8)
        ax.text(i, 0.30, f'n={v.size}\nmed {np.median(v):.1f}', ha='center',
                va='bottom', fontsize=7.4, color=GREY)
        # say what the uncorrected number would have been, where it differs
        raw = np.median([s_['r90'] for s_ in sessions if s_['cls'] == c])
        if abs(raw - np.median(v)) > 0.05:
            ax.text(i, 0.155, f'{raw:.1f} if gap blocks\nare left in',
                    ha='center', va='bottom', fontsize=6.8, color=C_HOT,
                    linespacing=1.25)

    ax.axhspan(0.1, ROW_PITCH_UM, color=C_ROW, alpha=.07, lw=0)
    ax.axhline(ROW_PITCH_UM, color=C_ROW, lw=1.0, ls='--')
    ax.axhline(KS4_HIGH_DRIFT_UM, color=C_HOT, lw=1.0, ls=':')
    ax.text(-0.42, ROW_PITCH_UM * 1.08, 'one 20 µm row', ha='left',
            va='bottom', fontsize=7.4, color=C_ROW)
    ax.text(-0.42, KS4_HIGH_DRIFT_UM * 1.08, "KS4's high-drift condition, 37 µm",
            ha='left', va='bottom', fontsize=7.4, color=C_HOT)
    ax.set_yscale('log')
    ax.set_xticks(range(3)); ax.set_xticklabels([CLS[c] for c in (0, 1, 2)],
                                                fontsize=8)
    ax.set_ylabel('drift, range$_{90}$ (µm)', fontsize=8.5, color=INK)
    ax.set_title(f'{len(sessions)} sessions.  Median {np.median(allv):.1f} µm; '
                 f'{100 * np.mean(allv < ROW_PITCH_UM):.0f}% below one row pitch.\n'
                 'These recordings barely reach the range the drift machinery\n'
                 'was built for — about the recordings, not about KS4.',
                 fontsize=8.8, loc='left', color=INK, pad=7, linespacing=1.4)
    _axes(ax)
    return _finish(fig, out_path)


def make_cost_panel(cost, out_path):
    """D: what drift costs the over-merge test, and why it is not fixable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.7),
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=.085, right=.985, top=.775, bottom=.155, wspace=.34)

    # --- D(i): ground truth, drift the only variable
    lab = ['no drift', 'fast', 'slow']
    val = [cost['fp_none'], cost['fp_fast'], cost['fp_slow']]
    col = [C_UNIT, C_HOT, C_HOT]
    a1.bar(range(3), val, width=.6, color=col, lw=0)
    for i, v in enumerate(val):
        a1.text(i, v + .4, f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                color=col[i], weight='semibold')
    a1.set_xticks(range(3)); a1.set_xticklabels(lab, fontsize=8)
    a1.set_ylabel('over-merge false positives (%)', fontsize=8.5, color=INK)
    a1.set_ylim(0, max(val) * 1.3)
    a1.set_title('Ground truth, drift the only variable',
                 fontsize=8.8, loc='left', color=INK, pad=7)
    _axes(a1)

    # --- D(ii): why no correction is available
    rem = cost['remedies']
    names = [r['name'] for r in rem]
    fp = np.array([r['fp'] for r in rem], float)
    det = np.array([r['detect'] for r in rem], float)
    yy = np.arange(len(rem))
    a2.barh(yy - .19, fp, height=.34, color=C_HOT, lw=0, label='FP, slow drift')
    a2.barh(yy + .19, det, height=.34, color=C_ROW, lw=0,
            label='two-unit detection')
    for y_, f_, d_ in zip(yy, fp, det):
        a2.text(f_ + 1.2, y_ - .19, f'{f_:.1f}%', va='center', fontsize=7.4,
                color=C_HOT)
        a2.text(d_ + 1.2, y_ + .19, f'{d_:.1f}%', va='center', fontsize=7.4,
                color=C_ROW)
    a2.set_yticks(yy); a2.set_yticklabels(names, fontsize=8)
    a2.invert_yaxis()
    a2.set_xlim(0, 100)
    a2.set_xlabel('%', fontsize=8.5, color=INK)
    a2.legend(fontsize=7.4, frameon=False, loc='lower right')
    a2.set_title('Every remedy trades the false positives\n'
                 'for the signal — drift and multi-unit structure\n'
                 'are confounded in the same features',
                 fontsize=8.8, loc='left', color=INK, pad=7, linespacing=1.4)
    _axes(a2)
    return _finish(fig, out_path)


def make_algorithm_panel(ks_dir, out_path, batch_s=2.0):
    """A: what KS4 computes, and where it stops being a measurement.

    Left, the probe: recorded sites against depth, with the blocks KS4 actually
    placed and how many channels each one contains. Right, the trace each of
    those blocks produced, offset so they can be told apart.

    THE POINT IS THE PAIRING. KS4 splits the DEPTH-BIN INDEX range into equal
    blocks with no reference to the channel map, so on a sparse or multi-bank
    probe a block can hold almost no channels -- here 2, 1, 1 and 6 against
    ~150 in the populated blocks -- and still emit a trace that looks like all
    the others. Panel B gives those same blocks' reading when the true drift is
    known to be zero.

    TWO THINGS THE FIRST VERSION GOT WRONG, both worth not repeating: it drew
    the block centres as a linspace over the depth span instead of reading the
    ones KS4 used, and it overlaid all nine traces, which on a session whose
    drift is mostly fast noise is a solid block of colour.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from unitassay.drift.gap_blocks import detect, _load
    from unitassay.drift.metrics import range_90
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    info = detect(ks_dir)
    d, yblk, y = _load(ks_dir)          # KS4's OWN block centres, not a guess
    is_gap = np.array(info['is_gap_block'], bool)
    n_in = np.array(info['channels_per_block'], int)
    step = info['block_spacing_um']
    nb = d.shape[1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.6),
                                 gridspec_kw=dict(width_ratios=[1, 2.35]))
    fig.subplots_adjust(left=.095, right=.975, top=.795, bottom=.125, wspace=.34)

    a1.scatter(np.full(y.size, .30), y, s=9, marker='_', color='#8A8F98',
               linewidths=.7)
    for i in range(nb):
        col = C_HOT if is_gap[i] else C_UNIT
        a1.add_patch(plt.Rectangle((.52, yblk[i] - step / 2), .34, step,
                                   facecolor=col,
                                   alpha=.85 if is_gap[i] else .30,
                                   edgecolor='white', lw=.6))
        a1.text(.90, yblk[i], f'{i}: {n_in[i]} ch', fontsize=6.6, va='center',
                ha='left', color=col,
                weight='bold' if is_gap[i] else 'normal')
    a1.set_xlim(0, 1.45)
    a1.set_ylim(float(y.min()) - 60, float(y.max()) + 60)
    a1.set_xticks([.30, .69])
    a1.set_xticklabels(['sites', 'blocks'], fontsize=7.5)
    a1.set_ylabel('depth along the probe (µm)', fontsize=8.5, color=INK)
    a1.set_title('the probe, and the blocks' + chr(10) + 'KS4 placed on it',
                 fontsize=8.8, loc='left', color=INK, pad=7, linespacing=1.35)
    _axes(a1)

    t = np.arange(d.shape[0]) * batch_s / 60.0
    centred = d - np.median(d, axis=0, keepdims=True)
    off = 2.2 * float(np.percentile(np.abs(centred), 97))
    for i in range(nb):
        g = bool(is_gap[i])
        a2.plot(t, centred[:, i] + i * off, lw=.75,
                color=C_HOT if g else C_UNIT, alpha=.95 if g else .8)
        a2.text(t[-1] * 1.012, i * off, str(i), fontsize=6.8, va='center',
                ha='left', color=C_HOT if g else C_UNIT,
                weight='bold' if g else 'normal')
    a2.set_xlim(t[0] - t[-1] * .01, t[-1] * 1.07)
    a2.set_yticks([0, (nb - 1) * off])
    a2.set_yticklabels(['block 0', 'block ' + str(nb - 1)], fontsize=7.5)
    a2.plot([t[-1] * 1.048, t[-1] * 1.048], [0, off], color=INK, lw=1.4)
    a2.text(t[-1] * 1.055, off / 2, '%.0f µm' % off, fontsize=6.8,
            va='center', ha='left', color=INK)
    a2.set_xlabel('time (min)', fontsize=8.5, color=INK)
    r_all = float(range_90(d))
    r_pop = float(range_90(d[:, ~is_gap])) if (~is_gap).any() else float('nan')
    a2.set_title('one trace per block - %d of %d hold almost no channels'
                 % (int(is_gap.sum()), nb) + chr(10) +
                 'range$_{90}$ %.1f µm with them, %.1f µm without'
                 % (r_all, r_pop),
                 fontsize=8.8, loc='left', color=INK, pad=7, linespacing=1.35)
    _axes(a2)
    return _finish(fig, out_path)
