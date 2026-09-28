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


def make_panel_a(steps, group, footer, out_path, width_in=15.0, height_in=3.5):
    """`steps` is a list of dicts: name, what, time, res, kind."""
    plt = _style()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    n = len(steps)
    gap = 0.026
    bw = (1.0 - 0.030 * 2 - gap * (n - 1)) / n
    y0, bh = 0.300, 0.440
    xs = [0.030 + i * (bw + gap) for i in range(n)]

    # the band naming OUR half of the pipeline, drawn first so it sits behind
    g0, g1 = group['first'], group['last']
    gx0, gx1 = xs[g0] - 0.011, xs[g1] + bw + 0.011
    ax.add_patch(FancyBboxPatch((gx0, y0 - 0.052), gx1 - gx0, bh + 0.128,
                                boxstyle='round,pad=0.004,rounding_size=0.012',
                                facecolor=BAND, edgecolor=UA_EDGE, lw=1.0,
                                linestyle=(0, (5, 3)), zorder=1))
    ax.text((gx0 + gx1) / 2, y0 + bh + 0.090, group['label'], ha='center',
            va='bottom', fontsize=12.5, weight='bold', color=UA_EDGE, zorder=5)

    C = {'raw': (RAW_EDGE, RAW_FILL), 'up': (UP_EDGE, UP_FILL),
         'ua': (UA_EDGE, UA_FILL), 'out': (OUT_EDGE, OUT_FILL)}
    for x, s in zip(xs, steps):
        edge, fill = C[s['kind']]
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh,
                                    boxstyle='round,pad=0.005,rounding_size=0.014',
                                    facecolor=fill, edgecolor=edge, lw=1.5,
                                    zorder=3))
        cx = x + bw / 2
        ax.text(cx, y0 + bh - 0.062, s['name'], ha='center', va='top',
                fontsize=11.5, weight='bold', color=edge, zorder=4)
        ax.text(cx, y0 + bh - 0.150, s['what'], ha='center', va='top',
                fontsize=8.6, color=INK, linespacing=1.45, zorder=4)
        # the rule separates WHAT IT DOES from WHAT IT COSTS, so a reader
        # scanning only for cost can find it in the same place every time
        ax.plot([x + 0.016, x + bw - 0.016], [y0 + 0.176] * 2, '-',
                color=edge, lw=0.8, alpha=.45, zorder=4)
        ax.text(cx, y0 + 0.146, s['time'], ha='center', va='top',
                fontsize=9.6, weight='bold', color=edge, zorder=4)
        ax.text(cx, y0 + 0.088, s['res'], ha='center', va='top',
                fontsize=7.8, color=MUTED, linespacing=1.5, zorder=4)

    for i in range(n - 1):
        x_from, x_to = xs[i] + bw, xs[i + 1]
        ax.add_patch(FancyArrowPatch((x_from + 0.004, y0 + bh / 2),
                                     (x_to - 0.004, y0 + bh / 2),
                                     arrowstyle='-|>', mutation_scale=15,
                                     color=GREY, lw=1.6, zorder=2))

    ax.text(0.030, 0.098, footer, ha='left', va='top', fontsize=8.6,
            color=MUTED, linespacing=1.65, zorder=4)

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
