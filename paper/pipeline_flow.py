"""Figure 5 panel A: the pipeline as boxes and arrows, with what each step costs.

EVERY NUMBER COMES FROM ONE RUN OF ONE PIPELINE. CatGT and Kilosort are read
from metadata/logs/catgt_ks4_pipeline_resources.jsonl, the per-step resource
log of the 147-session deployment; layers 1 and 2 from the `timing` block of
that same deployment's curated.json. So the four boxes describe the same
sessions on the same host, which is what makes their shares comparable.

Do NOT reach for resort_timing_stream{A,B}.json here. Those cover a
218-session superset on a different arm, and quoting them beside the
deployment numbers mixes corpora -- they give Kilosort 15.4 min against this
log's 11.2 min for the run the rest of the paper describes.

RESOURCES ARE ATTRIBUTED, NOT JUST OBSERVED. The 10.4 GB GPU peak belongs to
Kilosort; the same field is non-zero during layer 1 only because the card had
not been released yet -- layer 1's own GPU utilisation is 0%, so it is
reported as CPU-only rather than inheriting a number it never used.
"""
INK, MUTED, GREY = '#0D1B4B', '#546E7A', '#9aa7b4'
UP_EDGE, UP_FILL = '#8794A5', '#F1F3F6'      # upstream: someone else's code
UA_EDGE, UA_FILL = '#14213D', '#E8ECF3'      # ours
OUT_EDGE, OUT_FILL = '#12695A', '#E3F0EC'
RAW_EDGE, RAW_FILL = '#5C6B7A', '#FFFFFF'    # an input, not a processing step
BAND = '#F7F9FC'

# MINIMUM 12 pt ANYWHERE ON THE PANEL (Agrita, 2026-09-28). Sizes are named
# rather than written at the call site so the floor is checkable in one place,
# and asserted below so a later tweak cannot quietly go under it.
FS_NAME, FS_WHAT, FS_TIME, FS_RNG, FS_RES = 15, 12, 13.5, 12, 12
FS_GROUP, FS_FOOT = 16, 12
assert min(FS_NAME, FS_WHAT, FS_TIME, FS_RNG, FS_RES, FS_GROUP, FS_FOOT) >= 12


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'Nimbus Sans',
                            'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42})
    return plt


DEFAULT_SIZES = dict(name=FS_NAME, what=FS_WHAT, time=FS_TIME, rng=FS_RNG,
                     res=FS_RES, group=FS_GROUP)


def draw_pipeline_row(ax, steps, group, sizes=None, top=0.860, bh=0.505,
                      lw_box=1.6, arrow_scale=17):
    """Draw the boxes-and-arrows row into an EXISTING axes (0-1 coordinates).

    Split out of make_panel_a (2026-10-01) so the merged Figure 6 can put the
    pipeline in its top row at the figure's own font sizes and width, instead
    of stacking the 19-inch standalone panel. ONE drawing implementation:
    make_panel_a calls this.

    `steps`: dicts with name, what, time, rng, res, kind ('raw', 'up', 'ua',
    'out'). `group`: dict(first, last, label) -- the dashed UnitAssay band.
    `sizes`: font sizes in pt for name/what/time/rng/res/group.
    Returns (xs, bw, y0): box lefts, box width, box bottom -- for annotation.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    sz = dict(DEFAULT_SIZES, **(sizes or {}))
    fig = ax.figure
    ax_h_in = ax.get_position().height * fig.get_size_inches()[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    def lh(fs, mult=1.0):
        """One line of `fs` pt, as a fraction of the axes height."""
        return (fs / 72.0) / ax_h_in * mult

    n = len(steps)
    gap = 0.020
    bw = (1.0 - 0.022 * 2 - gap * (n - 1)) / n
    xs = [0.022 + i * (bw + gap) for i in range(n)]
    y0 = top - bh

    g0, g1 = group['first'], group['last']
    gx0, gx1 = xs[g0] - 0.009, xs[g1] + bw + 0.009
    ax.add_patch(FancyBboxPatch((gx0, y0 - 0.042), gx1 - gx0, bh + 0.104,
                                boxstyle='round,pad=0.004,rounding_size=0.012',
                                facecolor=BAND, edgecolor=UA_EDGE, lw=1.1,
                                linestyle=(0, (5, 3)), zorder=1))
    ax.text((gx0 + gx1) / 2, y0 + bh + 0.074, group['label'], ha='center',
            va='bottom', fontsize=sz['group'], weight='bold', color=UA_EDGE,
            zorder=5)

    C = {'raw': (RAW_EDGE, RAW_FILL), 'up': (UP_EDGE, UP_FILL),
         'ua': (UA_EDGE, UA_FILL), 'out': (OUT_EDGE, OUT_FILL)}
    for x, s in zip(xs, steps):
        edge, fill = C[s['kind']]
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh,
                                    boxstyle='round,pad=0.005,rounding_size=0.014',
                                    facecolor=fill, edgecolor=edge, lw=lw_box,
                                    zorder=3))
        cx, y = x + bw / 2, y0 + bh - lh(sz['name'], 1.5)

        def put(txt, fs, colour, weight='normal', alpha=1.0, pad=0.55):
            nonlocal y
            ax.text(cx, y, txt, ha='center', va='top', fontsize=fs,
                    weight=weight, color=colour, alpha=alpha,
                    linespacing=1.4, zorder=4)
            y -= lh(fs, 1.4) * txt.count('\n') + lh(fs, 1.0 + pad)

        put(s['name'], sz['name'], edge, 'bold', pad=0.35)
        put(s['what'], sz['what'], INK, pad=0.60)
        # the rule separates WHAT IT DOES from WHAT IT COSTS, so a reader
        # scanning only for cost finds it in the same place in every box
        ax.plot([x + 0.014, x + bw - 0.014], [y + lh(sz['time'], 0.45)] * 2, '-',
                color=edge, lw=0.9, alpha=.45, zorder=4)
        put(s['time'], sz['time'], edge, 'bold', pad=0.30)
        put(s['rng'], sz['rng'], edge, alpha=.85, pad=0.45)
        put(s['res'], sz['res'], MUTED)

    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + bw + 0.003, y0 + bh / 2),
                                     (xs[i + 1] - 0.003, y0 + bh / 2),
                                     arrowstyle='-|>', mutation_scale=arrow_scale,
                                     color=GREY, lw=1.8, zorder=2))
    return xs, bw, y0


def make_panel_a(steps, group, footer, out_path, width_in=19.0, height_in=5.0):
    """`steps` is a list of dicts: name, what, time, rng, res, kind.

    `time` is the median and `rng` the spread across sessions, on their own
    lines. A median with no spread beside it invites the reader to treat one
    number as the cost; these sessions run 2 to 36 minutes, and every step
    scales with that.

    Text is stacked from a cursor rather than at hand-tuned offsets, so the
    12 pt floor can be raised again without re-tuning six positions by eye.
    """
    plt = _style()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    xs, bw, y0 = draw_pipeline_row(ax, steps, group)

    ax.text(0.022, y0 - 0.090, footer, ha='left', va='top', fontsize=FS_FOOT,
            color=MUTED, linespacing=1.7, zorder=4)

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
