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

LAYER 1 IS SHOWN, because it is an input to every branch below -- but NOT as
a gate in series. All 53,346 clusters enter layer 2 whatever layer 1 said; its
verdict is carried alongside and is what sorts the terminus. Drawing it as a
first gate would say that layer 2 only sees what layer 1 passed, which is the
opposite of how the two layers work.

ONE PASS OUTCOME. The 3,978 that leave at NEIGHBOUR are passes, so they end in
the same box as the 20,561 that reach the end clean; splitting them into a
mid-flow branch and a separate total left one of them dangling in the margin.
Each outcome box now says what it is made of.

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
    '7-clean': 'No rule found anything,\nand layer 1 passed it.',
    '8-fall-through': 'No rule explains it,\nbut layer 1 rejected it.',
}
# WHAT EACH STAGE MEASURES. Agrita asked which metric decides identity; the
# boxes named their rule ids and never said. Each of these is the statistic the
# rule actually reads, with its shipped threshold, so the figure answers the
# question a reader will have at every box.
METRIC = {
    '1-identity': 'overlap > 50%  AND  |modal lag| > 0.2 ms',
    '2-refractory': 'ISI violation ratio, burst-corrected',
    '3-composition': "Hartigan's dip on the LDA projection, p < 0.05",
    '4-contamination': 'L-ratio against the reference channel',
    '5-neighbour': 'L-ratio in shared-channel PC space,\nCCG excess and refractory deficit',
    '7-clean': '', '8-fall-through': '', '6-unresolved': '',
}
VERDICT = {'1-identity': 'duplicate', '2-refractory': 'over-split',
           '3-composition': 'over-merge / over-split',
           '4-contamination': 'contaminated',
           '5-neighbour': 'not separable', '6-unresolved': 'no verdict',
           '7-clean': 'clean', '8-fall-through': 'unattributed'}


def make_decision_flow(flow, out_path, width_in=7.6, height_in=11.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    by = {s['stage']: s for s in flow['stages']}
    tests = [by[k] for k in ('1-identity', '2-refractory', '3-composition',
                             '4-contamination', '5-neighbour')]
    clean, fall = by['7-clean'], by['8-fall-through']
    total = flow['total']
    rescued = flow['rescued']
    attributed = flow['n_fail'] - fall['fail']

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    TOP = 0.905
    TESTS_BOT = 0.375
    Y_OUT = 0.135
    step = (TOP - TESTS_BOT) / len(tests)
    bx, bw = 0.255, 0.40
    bh = step * 0.72

    # --- layer 1, which every cluster has already been through
    ax.add_patch(FancyBboxPatch((bx - 0.02, TOP + 0.034), bw + 0.04, 0.052,
                                boxstyle='round,pad=0.005,rounding_size=0.010',
                                facecolor='#F4F6FA', edgecolor='#9AA3AE',
                                lw=0.9, zorder=3))
    ax.text(bx + bw / 2, TOP + 0.072, 'LAYER 1  (screen)', ha='center',
            va='center', fontsize=7.6, weight='bold', color='#5B6470',
            zorder=4)
    ax.text(bx + bw / 2, TOP + 0.052,
            f"{total:,} clusters  ·  passed {flow['screen_pass']:,}"
            f"  ·  rejected {flow['screen_fail']:,}",
            ha='center', va='center', fontsize=7.0, color=GREY, zorder=4)
    ax.text(bx + bw / 2 + 0.014, TOP + 0.016,
            'every cluster enters layer 2,\nwhatever layer 1 said',
            ha='left', va='center', fontsize=6.6, style='italic',
            color=GREY, linespacing=1.3)
    ax.annotate('', xy=(bx + bw / 2, TOP - step * 0.5 + bh / 2),
                xytext=(bx + bw / 2, TOP + 0.034),
                arrowprops=dict(arrowstyle='-|>', color=C_BOX_EDGE, lw=1.1))

    for i, s in enumerate(tests):
        yc = TOP - step * (i + 0.5)
        ax.add_patch(FancyBboxPatch((bx, yc - bh / 2), bw, bh,
                                    boxstyle='round,pad=0.006,rounding_size=0.012',
                                    facecolor=C_BOX, edgecolor=C_BOX_EDGE,
                                    lw=1.0, zorder=3))
        ax.text(bx + bw / 2, yc + bh / 2 - 0.009, s['label'].upper(),
                ha='center', va='top', fontsize=7.6, weight='bold',
                color=C_BOX_EDGE, zorder=4)
        ax.text(bx + bw / 2, yc + 0.004, SHORT_Q[s['stage']], ha='center',
                va='center', fontsize=8.6, color=INK, linespacing=1.35,
                zorder=4)
        ax.text(bx + bw / 2, yc - bh / 2 + 0.023, METRIC[s['stage']],
                ha='center', va='bottom', fontsize=6.6, color=C_BOX_EDGE,
                alpha=.85, linespacing=1.25, zorder=4)
        ax.text(bx + bw / 2, yc - bh / 2 + 0.006,
                '   '.join(f'rule {k}  {v:,}' for k, v in s['subs'].items()),
                ha='center', va='bottom', fontsize=6.4, color=GREY, zorder=4)

        if s['fail']:
            ax.annotate('', xy=(bx + bw + 0.072, yc), xytext=(bx + bw, yc),
                        arrowprops=dict(arrowstyle='-|>', color=C_FAIL, lw=1.3))
            ax.text(bx + bw + 0.086, yc + 0.006, f"FAIL  {s['fail']:,}",
                    ha='left', va='bottom', fontsize=8.6, color=C_FAIL,
                    weight='semibold')
            ax.text(bx + bw + 0.086, yc - 0.006,
                    f"{100 * s['fail'] / total:.1f}%  {VERDICT[s['stage']]}",
                    ha='left', va='top', fontsize=6.9, color=C_FAIL)
        nxt = (yc - step) + bh / 2 if i < len(tests) - 1 else Y_OUT + 0.085
        ax.annotate('', xy=(bx + bw / 2, nxt), xytext=(bx + bw / 2, yc - bh / 2),
                    arrowprops=dict(arrowstyle='-|>', color=C_BOX_EDGE, lw=1.1))
        # ENTERING MINUS REJECTED, not minus everything that left. The only
        # rule that also passes is NEIGHBOUR, and its 3,978 are carried down
        # to the single PASS outcome instead of branching off in the margin --
        # Agrita: one pass, at the end. So the number on the arrow is what no
        # rule has rejected, and at NEIGHBOUR that is 25,265, which is
        # 20,561 clean + 3,978 passed there + 726 nothing could explain.
        ax.text(bx + bw / 2 + 0.012, (yc - bh / 2 + nxt) / 2,
                f"{s['entering'] - s['fail']:,}", ha='left', va='center',
                fontsize=7, color=GREY)
    ax.text(bx + bw / 2 + 0.012, Y_OUT + 0.100, 'the rules stop here',
            ha='left', va='bottom', fontsize=7.2, style='italic', color=GREY)

    # --- two outcomes, and they ARE the totals
    for (col, side, verdict, n, parts) in (
            (C_PASS, -1, 'PASS', flow['n_pass'],
             f"clean {clean['pas']:,}\npassed at NEIGHBOUR {rescued:,}\n"
             f"(every one rejected by layer 1)"),
            (C_FAIL, +1, 'FAIL', flow['n_fail'],
             f"a rule named the fault {attributed:,}\n"
             f"no rule could explain it {fall['fail']:,}\n"
             f"(layer 1 had rejected those)")):
        cx = bx + bw / 2 + side * 0.195
        ax.add_patch(FancyBboxPatch((cx - 0.176, Y_OUT - 0.082), 0.352, 0.146,
                                    boxstyle='round,pad=0.006,rounding_size=0.012',
                                    facecolor='white', edgecolor=col, lw=1.3,
                                    zorder=3))
        ax.text(cx, Y_OUT + 0.036, f'{verdict}  {n:,}', ha='center',
                va='center', fontsize=11.5, weight='bold', color=col, zorder=4)
        ax.text(cx, Y_OUT + 0.014, f'{100 * n / total:.1f}% of clusters',
                ha='center', va='center', fontsize=7.2, color=GREY, zorder=4)
        ax.text(cx, Y_OUT - 0.038, parts, ha='center', va='center',
                fontsize=7.0, color=col, alpha=.9, linespacing=1.5, zorder=4)
        ax.annotate('', xy=(cx, Y_OUT + 0.064),
                    xytext=(bx + bw / 2, Y_OUT + 0.085),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2,
                                    connectionstyle='arc3,rad=%.2f' % (0.14 * side)))

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
