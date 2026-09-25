"""Figure 8 panel A: layer 2 as a sieve.

The stream runs left to right, its thickness proportional to how many clusters
are still undecided. Each rule takes a bite out of it: rejections fall away
below, and the one rule that can also pass -- 5a, the only two-sided gate in
the flow -- sends a tributary up. What emerges on the right is what passed.

WHY A SIEVE AND NOT A FLOWCHART. Ten of the eleven sub-rules can only reject.
Drawn as a decision tree they would each get a diamond and two branches, of
which one is always empty, and the picture would imply a symmetry the rules do
not have. The thickness of the stream is also the point: a reader should see
that contamination takes the largest bite and that most of what survives does
so by reaching the end untouched.

Rules that never fire are drawn, greyed, at zero. Omitting them would make the
rule set look smaller than the code, and someone reading the repository would
find two rules that the figure does not mention.
"""
import numpy as np

# Blue is what survives, magenta what is taken out -- the same roles the rows
# use, so a reader arrives at the rows already knowing which is which.
C_STREAM = '#6175B6'
C_STREAM_EDGE = '#3953A4'
C_FAIL = '#CE4679'
C_PASS = '#3953A4'
C_DEAD = '#C7CCD4'
INK = '#111418'
GREY = '#5B6470'


SHORT = {'1': 'already counted', '2': 'refractory ratio',
         '3a': 'two populations', '3b': 'time-ordered split',
         '4a': 'reference channel', '4b': 'evidence incomplete',
         '5a': 'separable from\nneighbours', '5b': 'coincidence excess',
         '5c': 'refractory deficit', '5d': 'evidence incomplete',
         '6': 'no verdict', '7': 'all rules passed', '8': 'unattributed'}


def make_sieve(flow, out_path, width_in=15.2, height_in=4.0):
    """Draw the sieve.

    THE PASS BAND ACCUMULATES ALONG THE TOP. A first version tapered the
    stream and wrote the total on the right, which made the 20,561 clusters
    rule 7 passes vanish: the stream shrank to 726 and a PASS figure appeared
    beside it from nowhere. Passes now leave upward into a band that grows,
    so the two things a reader counts -- what is taken out below, what is kept
    above -- are both visible and both add up.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    st, total = flow['stations'], flow['total']
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)

    n = len(st)
    x0, x1 = 0.075, 0.885
    xs = np.linspace(x0, x1, n + 1)
    TH = 0.145                       # stream thickness at full width
    Y = 0.50                         # stream top edge -- it hangs downward
    GAP = 0.055                      # between stream and pass band

    def h(c):
        return TH * c / total

    kept = 0
    for i, s in enumerate(st):
        a, b = xs[i], xs[i + 1]
        # undecided stream, hanging below a fixed top edge
        ax.add_patch(Polygon([(a, Y), (b, Y), (b, Y - h(s['remaining'])),
                              (a, Y - h(s['entering']))], closed=True,
                             facecolor=C_STREAM, edgecolor=C_STREAM_EDGE,
                             lw=0.4, alpha=.55, zorder=2))
        # pass band, growing above it
        k0 = kept
        kept += s['pas']
        if kept:
            ax.add_patch(Polygon([(a, Y + GAP), (b, Y + GAP),
                                  (b, Y + GAP + h(kept)),
                                  (a, Y + GAP + h(k0))], closed=True,
                                 facecolor=C_PASS, edgecolor='none',
                                 alpha=.80, zorder=2))

    # station labels, staggered so long names do not collide
    top = Y + GAP + h(total) + 0.02
    for i, s in enumerate(st):
        xm = xs[i + 1]
        dead = not s['fires']
        tier = 0.0 if i % 2 == 0 else 0.105
        ax.plot([xm, xm], [Y - h(s['remaining']) - 0.005, top + tier - 0.015],
                color=C_DEAD, lw=0.45, zorder=1)
        ax.text(xm, top + tier, s['rule'], ha='center', va='bottom',
                fontsize=9.5, weight='bold', color=C_DEAD if dead else INK)
        ax.text(xm, top + tier + 0.048, SHORT[s['rule']], ha='center',
                va='bottom', fontsize=6.6, linespacing=1.2,
                color=C_DEAD if dead else GREY)
        if dead:
            continue
        if s['fail']:
            # A CONNECTOR, NOT A QUANTITY. Sized by count and then capped so
            # the big ones fit, these read as roughly equal -- 382 looked much
            # like 5,818 -- which is a proportion the shape does not have. The
            # magnitude is already in the stream's taper and in the printed
            # number, so every tributary is now plainly the same width.
            half_w = 0.011
            drop = Y - h(s['entering'])
            ax.add_patch(Polygon([(xm - half_w, drop), (xm + half_w, drop),
                                  (xm + half_w * 0.55, drop - 0.085),
                                  (xm - half_w * 0.55, drop - 0.085)],
                                 closed=True, facecolor=C_FAIL,
                                 edgecolor='none', alpha=.85, zorder=3))
            lo = drop - 0.085
            ax.annotate('', xy=(xm, lo - 0.045), xytext=(xm, lo),
                        arrowprops=dict(arrowstyle='-|>', color=C_FAIL, lw=.9))
            ax.text(xm, lo - 0.062, f"{s['fail']:,}", ha='center', va='top',
                    fontsize=7.6, color=C_FAIL, weight='semibold')
            ax.text(xm, lo - 0.108, f"{100 * s['fail'] / total:.1f}%",
                    ha='center', va='top', fontsize=6.2, color=C_FAIL)
        if s['pas']:
            ax.text(xm - 0.004, Y + GAP + h(kept) + 0.012,
                    f"+{s['pas']:,} pass", ha='center', va='bottom',
                    fontsize=7.4, color=C_PASS, weight='semibold')

    ax.text(x0 - 0.01, Y - h(total) / 2, f"{total:,}\nclusters", ha='right',
            va='center', fontsize=9.5, color=INK, weight='semibold',
            linespacing=1.3)
    ax.text(x0 - 0.01, Y - h(total) - 0.045, f"{flow['n_sessions']} sessions",
            ha='right', va='top', fontsize=7, color=GREY)
    ax.text(x1 + 0.012, Y + GAP + h(total) / 2,
            f"PASS\n{flow['n_pass']:,}", ha='left', va='center',
            fontsize=10, color=C_PASS, weight='bold', linespacing=1.3)
    ax.text(x1 + 0.012, Y + GAP + 0.005,
            f"{100 * flow['n_pass'] / total:.1f}% of clusters", ha='left',
            va='bottom', fontsize=7, color=GREY)
    # with the rejections, not beside rule 8's drop, which it sat on top of
    ax.text(x1 + 0.012, 0.16, f"FAIL\n{flow['n_fail']:,}", ha='left',
            va='center', fontsize=9.5, color=C_FAIL, weight='bold',
            linespacing=1.3)
    ax.text(x1 + 0.012, 0.105,
            f"{100 * flow['n_fail'] / total:.1f}% of clusters", ha='left',
            va='center', fontsize=7, color=GREY)

    i5 = [s['rule'] for s in st].index('5a')
    ax.text(xs[i5 + 1], 0.012,
            'rule 5a is the only two-sided gate — every other rule can only reject',
            ha='center', va='bottom', fontsize=7, color=GREY)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
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
    return None
