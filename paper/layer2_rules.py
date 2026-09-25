"""Figure 8 -- layer 2 (`unitassay.curate`): the rules, and one row per outcome.

The figure is a vertical sieve (panel A) beside one row per error class, each
row drawn across six columns of evidence. Its claim is that no single metric
is sufficient, so a row earns its place by showing FIVE quiet columns and one
that fires.

Data functions are named for what they regenerate, so the figure can be rebuilt
from this module alone.

WHY SELECTION IS BY THE RULE THAT FIRED, not by re-testing the metrics.
Reconstructing a shipped rule to decide which column "should" fire is how the
layer-1 reference-channel panel got inverted: the reconstruction sought
something subtly different from the rule and the panel then argued the opposite
of the tool. Here the firing column is read off `rule_id` -- the tool's own
verdict -- and the raw metrics are used ONLY to check that the other columns
are quiet.

NOT EVALUATED IS NOT QUIET. A dip that was never run and a dip that ran and
found one mode both leave a column with no red in it, but only the second is
evidence. Candidates are ranked so that rows whose quiet columns were actually
tested come first, and `n_untested` is reported for every candidate.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from unitassay.curate.constants import (ABSENT_FRAME_FRAC_THRESH,
                                        DIP_PVAL_THRESH, LOW_SPIKE_THRESHOLD,
                                        ccg_z_threshold)

# The six columns, in the order they are drawn -- which follows the sieve, so
# that reading a row left to right is reading panel A top to bottom.
COLUMNS = ('waveform', 'isi', 'projection', 'amplitude', 'feature_space', 'ccg')

# Which column carries the evidence for each rule. `waveform` is context for
# every row and is never a firing column. 5d/7/8 reach a verdict without one
# rule's evidence dominating, so they have no firing column.
RULE_COLUMN = {
    '1': 'ccg',            # overlap and lag
    '2': 'isi',            # refractory ratio
    '3a': 'projection',    # bimodality -- the dip runs on the 1-D projection
    '3b': 'amplitude',     # time-ordered split
    '4a': 'feature_space',  # reference channel
    '4b': 'feature_space',
    '5a': 'feature_space',  # neighbour separability
    '5b': 'ccg',           # coincidence excess
    '5c': 'ccg',           # refractory deficit
    '5d': None, '6': None, '7': None, '8': None,
}

# The rows of the figure, top to bottom. Clean leads: without it the columns
# look like they fire on everything and the specificity half of the argument
# is invisible.
ROWS = ('clean', 'oversplit', 'mixed_cluster', 'duplicate',
        'intermittent_presence')


def _isfinite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def column_state(row, k=1):
    """Per column: 'fires', 'quiet', or 'untested'.

    Each test is the SHIPPED one -- `ccg_z_threshold` is the calibrated curve,
    not a fixed z, and the absent-frame threshold is derived from the shipped
    isolation t-ratio rather than guessed.
    """
    st = {'waveform': 'quiet'}   # context column, never fires

    ok = row.get('passes_isi_independent')
    st['isi'] = 'untested' if ok is None else ('quiet' if ok else 'fires')

    dp = row.get('_dip_pval')
    st['projection'] = ('untested' if not _isfinite(dp)
                        else ('fires' if dp < DIP_PVAL_THRESH else 'quiet'))

    af = row.get('absent_frame_frac')
    st['amplitude'] = ('untested' if not _isfinite(af)
                       else ('fires' if af > ABSENT_FRAME_FRAC_THRESH else 'quiet'))

    sep = row.get('between_cluster_confirmed_separated')
    st['feature_space'] = ('untested' if sep is None
                           else ('quiet' if sep else 'fires'))

    acc = row.get('ccg_accusers')
    if row.get('overlap_pairs'):
        st['ccg'] = 'fires'
    elif acc is None:
        st['ccg'] = 'untested'
    elif not acc:
        st['ccg'] = 'quiet'
    else:
        usable = [a for a in acc if _isfinite(a.get('z')) and a.get('n_min')]
        if not usable:
            st['ccg'] = 'untested'
        elif any(a['z'] > ccg_z_threshold(a['n_min'], k) for a in usable):
            st['ccg'] = 'fires'
        else:
            st['ccg'] = 'quiet'
    return st


def exemplar_candidates(corpus_dir, min_spikes=LOW_SPIKE_THRESHOLD,
                        max_spikes=40000, per_row=12):
    """Candidate clusters for each row of Figure 8, best first.

    A candidate must be the modal, teachable case: the column its rule reads
    fires, and EVERY other column is quiet -- five normal-looking panels and
    one that does not. Ranked by how many of those quiet columns were actually
    evaluated, since an untested column teaches nothing.
    """
    corpus_dir = Path(corpus_dir)
    out = {r: [] for r in ROWS}
    for path in sorted(corpus_dir.glob('*/curated.json')):
        rec = json.load(open(path))
        for row in rec['rows']:
            ec = row.get('error_class')
            if ec not in out:
                continue
            n = row.get('n_spikes') or 0
            if not (min_spikes <= n <= max_spikes):
                continue
            rid = str(row.get('rule_id') or '')
            want = RULE_COLUMN.get(rid)
            st = column_state(row)
            fires = [c for c in COLUMNS if st[c] == 'fires']

            # THE FIRING COLUMN COMES FROM THE RULE, NOT FROM RE-TESTING.
            # A rule that fires early short-circuits the later evidence: every
            # one of rule 4a's clusters has `between_cluster_confirmed_separated
            # = None`, because rule 5 never ran on it. Deciding the firing
            # column from the metrics therefore made the single biggest rule
            # (4a, 21.3% of the corpus) unselectable. The rule's own verdict is
            # authoritative; the metrics only certify that the OTHER columns
            # are quiet.
            if ec == 'clean':
                want = None
            elif ec == 'intermittent_presence':
                # No layer-2 rule claims these -- all 102 fall through to rule
                # 8 -- so layer 1's presence test is the evidence and the
                # amplitude column is the firing one.
                want = 'amplitude'
            if ec != 'clean' and want is None:
                continue
            others = [c for c in COLUMNS if c != want]
            if any(st[c] == 'fires' for c in others):
                continue
            fires = [want] if want else []

            quiet = [c for c in others if st[c] == 'quiet' and c != 'waveform']
            untested = [c for c in others if st[c] == 'untested']
            out[ec].append({
                'session': rec['session'], 'clu_id': row.get('clu_id'),
                'ks_dir': rec['ks_dir'], 'n_spikes': n,
                'rule_id': rid, 'rule_label': row.get('rule_label'),
                'sub_type': row.get('sub_type'), 'decision': row.get('decision'),
                'dip_basis': row.get('dip_basis'),
                'fires': fires, 'n_quiet': len(quiet),
                'n_untested': len(untested), 'untested': untested,
            })
    for r in out:
        # Most evaluated-and-quiet columns first; then mid-sized clusters, which
        # draw better than either extreme.
        out[r].sort(key=lambda c: (-c['n_quiet'], c['n_untested'],
                                   abs(math.log10(max(c['n_spikes'], 1)) - 3.1)))
        out[r] = out[r][:per_row]
    return out


def write_exemplar_candidates(corpus_dir, out_json, **kw):
    """Regenerate the candidate shortlist Figure 8's rows are chosen from."""
    cand = exemplar_candidates(corpus_dir, **kw)
    out_json = Path(out_json)
    out_json.write_text(json.dumps(
        {'corpus': str(corpus_dir), 'columns': list(COLUMNS),
         'rows': list(ROWS), 'candidates': cand}, indent=1))
    return out_json
