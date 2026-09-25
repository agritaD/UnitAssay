"""Figure 8 panel A: the decision flow.

A cluster enters at the top and is asked one question per stage, in the order
curate() asks them. Answer one way and it leaves immediately with a verdict;
answer the other and it goes on to the next question. Reaching the bottom
untouched is itself a verdict -- "all rules passed".

BY STAGE, NOT BY SUB-RULE. A stage is a SCOPE: what the test could see. That
is what a reader following one cluster is actually deciding at each step, and
it keeps the flow to seven questions instead of thirteen. The sub-rules are
printed inside each box with their own counts, so nothing is hidden.

Only stage 5 can send a cluster BOTH ways. Every other stage either rejects or
passes the cluster along, which is why the exits are all on one side until
that point.
"""
import numpy as np

C_BOX = '#EDF1F8'
C_BOX_EDGE = '#3953A4'
C_FAIL = '#CE4679'
C_PASS = '#3953A4'
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


def make_decision_flow(flow, out_path, width_in=6.8, height_in=13.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})

    st = [s for s in flow['stages'] if s['fires'] or s['stage'] in
          ('6-unresolved',)]
    total = flow['total']
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    n = len(st)
    top, bot = 0.955, 0.075
    step = (top - bot) / n
    # The box sits in the middle with room either side: PASS leaves left,
    # FAIL leaves right, and both labels need space to be read. The first
    # layout gave the box 0.52 of the width and both exits ran off the page.
    bx, bw = 0.255, 0.40
    bh = step * 0.52

    ax.text(bx + bw / 2, top + 0.028, f"{total:,} clusters",
            ha='center', va='bottom', fontsize=11, weight='bold', color=INK)
    ax.text(bx + bw / 2, top + 0.012, f"{flow['n_sessions']} sessions",
            ha='center', va='bottom', fontsize=7.5, color=GREY)

    for i, s in enumerate(st):
        yc = top - step * (i + 0.5)
        dead = not s['fires']
        edge = C_DEAD if dead else C_BOX_EDGE
        ax.add_patch(FancyBboxPatch((bx, yc - bh / 2), bw, bh,
                                    boxstyle='round,pad=0.006,rounding_size=0.012',
                                    facecolor='white' if dead else C_BOX,
                                    edgecolor=edge, lw=1.0, zorder=3))
        # stage name on its own line above the question -- overlaid on it,
        # the two were unreadable
        ax.text(bx + bw / 2, yc + bh / 2 - 0.009, s['label'].upper(),
                ha='center', va='top', fontsize=7.4, weight='bold',
                color=C_DEAD if dead else C_BOX_EDGE, zorder=4)
        ax.text(bx + bw / 2, yc + 0.0015, SHORT_Q[s['stage']], ha='center',
                va='center', fontsize=8.4, color=C_DEAD if dead else INK,
                linespacing=1.35, zorder=4)
        subtxt = '   '.join(f'rule {k}  {v:,}' for k, v in s['subs'].items())
        ax.text(bx + bw / 2, yc - bh / 2 + 0.008,
                subtxt or 'never fires', ha='center', va='bottom',
                fontsize=6.4, color=C_DEAD if dead else GREY, zorder=4)

        # the exit, and what is left
        if s['fail']:
            ax.annotate('', xy=(bx + bw + 0.075, yc), xytext=(bx + bw, yc),
                        arrowprops=dict(arrowstyle='-|>', color=C_FAIL, lw=1.3))
            ax.text(bx + bw + 0.090, yc + 0.006, f"FAIL  {s['fail']:,}",
                    ha='left', va='bottom', fontsize=8.6, color=C_FAIL,
                    weight='semibold')
            ax.text(bx + bw + 0.090, yc - 0.006,
                    f"{100 * s['fail'] / total:.1f}%  {VERDICT[s['stage']]}",
                    ha='left', va='top', fontsize=6.9, color=C_FAIL)
        if s['pas']:
            ax.annotate('', xy=(bx - 0.075, yc), xytext=(bx, yc),
                        arrowprops=dict(arrowstyle='-|>', color=C_PASS, lw=1.3))
            ax.text(bx - 0.090, yc + 0.006, f"PASS  {s['pas']:,}", ha='right',
                    va='bottom', fontsize=8.6, color=C_PASS, weight='semibold')
            ax.text(bx - 0.090, yc - 0.006, f"{100 * s['pas'] / total:.1f}%",
                    ha='right', va='top', fontsize=6.9, color=C_PASS)

        if i < n - 1:
            y_from, y_to = yc - bh / 2, top - step * (i + 1.5) + bh / 2
            ax.annotate('', xy=(bx + bw / 2, y_to), xytext=(bx + bw / 2, y_from),
                        arrowprops=dict(arrowstyle='-|>', color=C_BOX_EDGE,
                                        lw=1.1))
            ax.text(bx + bw / 2 + 0.012, (y_from + y_to) / 2,
                    f"{s['remaining']:,}", ha='left', va='center',
                    fontsize=6.8, color=GREY)

    ax.text(bx + bw / 2, bot - 0.030,
            'Only NEIGHBOUR can send a cluster both ways;\n'
            'every other stage either rejects it or passes it on.',
            ha='center', va='top', fontsize=7.2, color=GREY, linespacing=1.4)

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
