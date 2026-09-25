"""Figure 8 panel A: the decision flow.

A cluster enters at the top and is asked one question per stage, in the order
curate() asks them. Answer one way and it leaves immediately with a verdict;
answer the other and it goes on to the next question. Reaching the bottom
untouched is itself a verdict -- "all rules passed".

BY STAGE, NOT BY SUB-RULE. A stage is a SCOPE: what the test could see. That
is what a reader following one cluster is actually deciding at each step, and
it keeps the flow to seven questions instead of thirteen. The sub-rules are
printed inside each box with their own counts, so nothing is hidden.

Only NEIGHBOUR can send a cluster both ways. Every other stage either rejects
it or passes it along, which is why the exits are on one side until then.

THE RULES STOP AT NEIGHBOUR. What follows is not another test: clusters that
survive all five reach a terminus and are sorted by what layer 1 already said
about them -- `screen_pass` makes them CLEAN, and what nothing has claimed is
FALL-THROUGH. Drawing fall-through BELOW clean, as the first version did,
implied a sixth test applied to clusters that had already passed, which does
not exist. They are siblings at the end.
"""
import numpy as np

C_BOX = '#EDF1F8'
C_BOX_EDGE = '#3953A4'
C_FAIL = '#CE4679'
C_PASS = '#0A8F40'          # green: what survives
C_DEAD = '#C7CCD4'
INK = '#111418'
GREY = '#5B6470'

SHORT_Q = {
    '1-identity': 'Are these spikes already\ncounted under another cluster?',
    '2-refractory': 'Could one neuron\nproduce these spikes?',
    '3-composition': 'Are there two populations\ninside it?',
    '4-contamination': 'Is it polluted by something\nthis cluster alone can see?',
    '5-neighbour': 'Is it separable from the\nneighbours it is tested against?',
    '6-unresolved': 'Could it be tested at all?',
    '7-clean': 'Nothing found.',
    '8-fall-through': 'Nothing claimed it.',
}
VERDICT = {'1-identity': 'duplicate', '2-refractory': 'over-split',
           '3-composition': 'over-merge / over-split',
           '4-contamination': 'contaminated',
           '5-neighbour': 'not separable', '6-unresolved': 'no verdict',
           '7-clean': 'clean', '8-fall-through': 'unattributed'}


def make_decision_flow(flow, out_path, width_in=7.4, height_in=13.2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    by = {s['stage']: s for s in flow['stages']}
    # the five questions; UNRESOLVED is dropped -- it never fires on this
    # corpus, and an empty box in a decision flow reads as a step a cluster
    # has to get past
    tests = [by[k] for k in ('1-identity', '2-refractory', '3-composition',
                             '4-contamination', '5-neighbour')]
    clean, fall = by['7-clean'], by['8-fall-through']
    total = flow['total']

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Explicit vertical budget. Deriving the terminus from the tests' lower
    # bound put the two outcome boxes on top of the NEIGHBOUR box.
    TOP = 0.962          # first question
    TESTS_BOT = 0.415    # last question's lower edge
    Y_TERM = 0.300       # the two outcome boxes
    Y_TOTAL = 0.135      # what the whole flow produced
    top, bot = TOP, TESTS_BOT
    step = (top - bot) / len(tests)
    bx, bw = 0.255, 0.40
    bh = step * 0.62

    ax.text(bx + bw / 2, top + 0.026, f"{total:,} clusters", ha='center',
            va='bottom', fontsize=11.5, weight='bold', color=INK)
    ax.text(bx + bw / 2, top + 0.012, f"{flow['n_sessions']} sessions",
            ha='center', va='bottom', fontsize=7.5, color=GREY)

    for i, s in enumerate(tests):
        yc = top - step * (i + 0.5)
        ax.add_patch(FancyBboxPatch((bx, yc - bh / 2), bw, bh,
                                    boxstyle='round,pad=0.006,rounding_size=0.012',
                                    facecolor=C_BOX, edgecolor=C_BOX_EDGE,
                                    lw=1.0, zorder=3))
        ax.text(bx + bw / 2, yc + bh / 2 - 0.009, s['label'].upper(),
                ha='center', va='top', fontsize=7.6, weight='bold',
                color=C_BOX_EDGE, zorder=4)
        ax.text(bx + bw / 2, yc + 0.002, SHORT_Q[s['stage']], ha='center',
                va='center', fontsize=8.6, color=INK, linespacing=1.35,
                zorder=4)
        ax.text(bx + bw / 2, yc - bh / 2 + 0.008,
                '   '.join(f'rule {k}  {v:,}' for k, v in s['subs'].items()),
                ha='center', va='bottom', fontsize=6.5, color=GREY, zorder=4)

        if s['fail']:
            ax.annotate('', xy=(bx + bw + 0.072, yc), xytext=(bx + bw, yc),
                        arrowprops=dict(arrowstyle='-|>', color=C_FAIL, lw=1.3))
            ax.text(bx + bw + 0.086, yc + 0.006, f"FAIL  {s['fail']:,}",
                    ha='left', va='bottom', fontsize=8.6, color=C_FAIL,
                    weight='semibold')
            ax.text(bx + bw + 0.086, yc - 0.006,
                    f"{100 * s['fail'] / total:.1f}%  {VERDICT[s['stage']]}",
                    ha='left', va='top', fontsize=6.9, color=C_FAIL)
        if s['pas']:
            ax.annotate('', xy=(bx - 0.072, yc), xytext=(bx, yc),
                        arrowprops=dict(arrowstyle='-|>', color=C_PASS, lw=1.3))
            ax.text(bx - 0.086, yc + 0.006, f"PASS  {s['pas']:,}", ha='right',
                    va='bottom', fontsize=8.6, color=C_PASS, weight='semibold')
            ax.text(bx - 0.086, yc - 0.006, f"{100 * s['pas'] / total:.1f}%",
                    ha='right', va='top', fontsize=6.9, color=C_PASS)

        nxt = (yc - step) + bh / 2 if i < len(tests) - 1 else Y_TERM + 0.085
        ax.annotate('', xy=(bx + bw / 2, nxt), xytext=(bx + bw / 2, yc - bh / 2),
                    arrowprops=dict(arrowstyle='-|>', color=C_BOX_EDGE, lw=1.1))
        ax.text(bx + bw / 2 + 0.012, (yc - bh / 2 + nxt) / 2,
                f"{s['remaining']:,}", ha='left', va='center', fontsize=7,
                color=GREY)

    # --- the terminus. Not a test: what survived all five is sorted by what
    #     layer 1 already said, so these two are siblings, not a sequence.
    ty = Y_TERM
    ax.text(bx + bw / 2 + 0.014, Y_TERM + 0.098, 'the rules stop here',
            ha='left', va='bottom', fontsize=7.4, style='italic', color=GREY)
    for (s, col, side, verdict) in ((clean, C_PASS, -1, 'PASS'),
                                    (fall, C_FAIL, +1, 'FAIL')):
        cx = bx + bw / 2 + side * 0.185
        ax.add_patch(FancyBboxPatch((cx - 0.158, ty - 0.055), 0.316, 0.100,
                                    boxstyle='round,pad=0.005,rounding_size=0.012',
                                    facecolor='white', edgecolor=col, lw=1.1,
                                    zorder=3))
        ax.annotate('', xy=(cx, ty + 0.045), xytext=(bx + bw / 2, Y_TERM + 0.085),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2,
                                    connectionstyle='arc3,rad=%.2f' % (0.12 * side)))
        n = s['pas'] or s['fail']
        ax.text(cx, ty + 0.023, s['label'].upper(), ha='center', va='center',
                fontsize=7.6, weight='bold', color=col, zorder=4)
        ax.text(cx, ty - 0.002, f'{verdict}  {n:,}', ha='center', va='center',
                fontsize=8.8, weight='semibold', color=col, zorder=4)
        ax.text(cx, ty - 0.034, SHORT_Q[s['stage']].replace('\n', ' '),
                ha='center', va='center', fontsize=6.6, color=GREY, zorder=4)

    # --- what the whole flow produced
    fy = Y_TOTAL
    ax.plot([0.10, 0.90], [fy + 0.048, fy + 0.048], color='#D8DCE6', lw=0.8)
    ax.text(bx + bw / 2 - 0.185, fy, f"PASS  {flow['n_pass']:,}", ha='center',
            va='center', fontsize=11, weight='bold', color=C_PASS)
    ax.text(bx + bw / 2 - 0.185, fy - 0.030,
            f"{100 * flow['n_pass'] / total:.1f}% of clusters", ha='center',
            va='center', fontsize=7.2, color=GREY)
    ax.text(bx + bw / 2 + 0.185, fy, f"FAIL  {flow['n_fail']:,}", ha='center',
            va='center', fontsize=11, weight='bold', color=C_FAIL)
    ax.text(bx + bw / 2 + 0.185, fy - 0.030,
            f"{100 * flow['n_fail'] / total:.1f}% of clusters", ha='center',
            va='center', fontsize=7.2, color=GREY)

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
