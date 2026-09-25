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
from pathlib import Path

import numpy as np

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


def rule_partner(row):
    """The cluster the RULE compared this one against, and why."""
    if row.get('redundant_of') is not None:
        return int(row['redundant_of']), 'counted under'
    if row.get('overlap_pairs'):
        q = row['overlap_pairs'][0]
        # 'other' here, 'clu' in ccg_accusers -- the two schemas differ
        pid = int(q.get('other', q.get('clu'))) if isinstance(q, dict) else int(q)
        return pid, 'overlapping'
    acc = [a for a in (row.get('ccg_accusers') or [])
           if isinstance(a, dict) and a.get('z') is not None]
    if acc:
        return int(max(acc, key=lambda a: a['z'])['clu']), 'accuser'
    return None, None


def layer1_neighbours(session_dir, clu_id):
    """The same-peak-channel clusters layer 1 referenced this one against.

    other_clu is stored as 1-INDEXED ms_ids -- compute_isolation builds them
    from `my_clus = [c + 1 ...]` and compares them against `good_clus_0 + 1`.
    Measured on 25 sessions / 23,158 references: decrementing gives a peak
    channel identical to the unit's 100.00% of the time, the raw value 32.3%,
    and the raw value points a cluster at ITSELF 2,727 times.
    """
    import h5py
    p = Path(session_dir) / 'isolation' / f'{clu_id + 1}.SortData.h5'
    if not p.exists():
        return set()
    try:
        with h5py.File(p, 'r') as h:
            keys = sorted(h.keys())
            if not keys or 'other_clu' not in h[keys[0]]:
                return set()
            raw = np.atleast_1d(h[keys[0]]['other_clu'][:]).astype(int)
    except Exception:
        return set()
    return {int(c) - 1 for c in raw if int(c) - 1 >= 0}


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

            # CONSISTENT WITH LAYER 1 -- which is the figure's point. The
            # cluster a rule accuses is not always one layer 1 referenced:
            # layer 1 only ever looks at the SAME peak channel, while layer 2
            # scans wider. Measured per class, the partner is a layer-1
            # neighbour for clean 95.2%, over-merge 93.9%, over-split 74.5%
            # -- but duplicates only 47.2%, with 39.6% on a DIFFERENT channel,
            # because re-detection is spatial. So the constraint is applied
            # everywhere EXCEPT duplicates, where forcing it would
            # misrepresent the mechanism; that row instead labels the channel
            # relation, which is exactly why layer 1 cannot see it.
            pid, _why = rule_partner(row)
            nb = layer1_neighbours(path.parent, row.get('clu_id'))
            # This describes THE CLUSTER THE ROW WILL ACTUALLY DRAW, not the
            # rule partner. Rules 4a and 8 name no partner, so the row falls
            # back to a layer-1 neighbour -- which is same-channel by
            # construction. Reporting False there put 'different peak channel'
            # on a panel that also said 'same channel' three lines above it.
            if pid is None:
                same_ch = True if nb else None
            else:
                same_ch = pid in nb
            if ec != 'duplicate' and pid is not None and not same_ch:
                continue

            quiet = [c for c in others if st[c] == 'quiet' and c != 'waveform']
            untested = [c for c in others if st[c] == 'untested']
            out[ec].append({
                'session': rec['session'], 'clu_id': row.get('clu_id'),
                'ks_dir': rec['ks_dir'], 'n_spikes': n,
                'rule_id': rid, 'rule_label': row.get('rule_label'),
                'sub_type': row.get('sub_type'), 'decision': row.get('decision'),
                'dip_basis': row.get('dip_basis'),
                'partner': pid, 'drawn_partner_same_channel': same_ch,
                'layer1_neighbours': sorted(nb),
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


# ---------------------------------------------------------------------------
# Over-splitting has TWO faces in real data, and the figure shows both.
#
# Figure 3A's over-split example has a refractory DIP at zero lag: one neuron's
# spikes cut in two, so the real cell's refractory period forbids an A spike
# and a B spike landing 1 ms apart. That is the mechanism the word names.
#
# Most clusters the taxonomy calls over-split do not look like that. Measured
# per rule over 20 sessions (dip ratio = mean |lag| <= 1 ms / mean 5-20 ms
# flanks), only rule 5c -- the refractory-deficit rule -- dips:
#
#     5c oversplit  median 0.57   30.0% dipping    4.0% peaked
#     7  clean      median 0.90    7.7%           12.5%
#     4a oversplit  median 0.96    6.8%           23.2%
#     3a oversplit  median 1.10    3.2%           34.8%
#     2  oversplit  median 1.39    1.4%           45.9%
#
# A zero-lag EXCESS is the duplicate signature, and rule 2's over-splits carry
# one 45.9% of the time against clean's 12.5%. Without ground truth those
# cannot be called over-split or duplicate -- which is the point the figure
# makes explicitly rather than hides by showing only the clean case.
#
# A zero-lag peak is not proof of duplication either: synchronous neurons make
# one too. That is exactly why the shipped criterion is the LAG (overlap > 50%
# AND |modal lag| > 0.2 ms), not the excess.

CCG_SHAPES = ('refractory_dip', 'coincident_excess', 'flat')


def ccg_shape(times_a, times_b, fs=30000.0):
    """(modal lag ms, dip ratio, shape, flank count) for one pair.

    Uses the SHIPPED window, bin and dip statistic, so the label here is the
    one the rule's own test would give.
    """
    from unitassay.curate.ccg import (CCG_BIN_MS, CCG_WINDOW_MS, compute_ccg,
                                      dip_verdict)
    if times_a is None or times_b is None or not len(times_a) or not len(times_b):
        return None, None, None, 0
    centers, counts = compute_ccg(np.asarray(times_a) * fs,
                                  np.asarray(times_b) * fs,
                                  window_ms=CCG_WINDOW_MS, bin_ms=CCG_BIN_MS,
                                  fs=fs)
    if not counts.size or counts.max() <= 0:
        return None, None, None, 0
    ratio, _label = dip_verdict(centers, counts)
    modal = float(centers[int(np.argmax(counts))])
    # AN EMPTY CENTRE IS NOT A REFRACTORY HOLE UNLESS THE FLANKS ARE FULL.
    # dip_ratio = 0.00 falls out of any pair too sparse to put counts in the
    # +-1 ms window, and the first pass ranked exactly those to the top -- with
    # modal lags at -22 and -17 ms, i.e. the correlogram's peak was noise. The
    # dip is only evidence when there is something for the centre to be a hole
    # IN, so the flanks carry the power requirement.
    flank = counts[(np.abs(centers) > 5) & (np.abs(centers) <= 20)]
    flank_total = int(flank.sum()) if flank.size else 0
    shape = ('refractory_dip' if (ratio < 0.3 and flank_total >= 60)
             else 'coincident_excess' if ratio > 1.5 else 'flat')
    return modal, float(ratio), shape, flank_total


def oversplit_by_ccg_shape(corpus_dir, candidates, max_eval=400, per_group=8):
    """Split the over-split candidates into the two faces, best first.

    Returns {'oversplit_dip': [...], 'oversplit_nodip': [...]}, each entry
    carrying modal_lag / dip_ratio / ccg_shape so the panel and the caption
    quote the same numbers.
    """
    from unitassay.screen.io.ks_loader import load_ks_dir
    corpus_dir = Path(corpus_dir)
    cache = {}

    def spikes(ks, clu):
        if ks not in cache:
            cache[ks] = load_ks_dir(ks, exclude_noise=False, load_pcs=False)
        sp = cache[ks]
        return sp.st[sp.clu == clu]

    out = {'oversplit_dip': [], 'oversplit_nodip': []}
    for c in candidates[:max_eval]:
        sess = corpus_dir / c['session']
        pid = c.get('partner')
        if pid is None:
            nb = sorted(c.get('layer1_neighbours') or [])
            pid = nb[0] if nb else None
        if pid is None:
            continue
        try:
            ta, tb = spikes(c['ks_dir'], c['clu_id']), spikes(c['ks_dir'], pid)
        except Exception:
            continue
        modal, ratio, shape, flank = ccg_shape(ta, tb)
        if shape is None:
            continue
        rec = dict(c, partner=pid, modal_lag_ms=modal, dip_ratio=ratio,
                   ccg_shape=shape, flank_counts=flank)
        if shape == 'refractory_dip':
            out['oversplit_dip'].append(rec)
        elif shape == 'coincident_excess':
            out['oversplit_nodip'].append(rec)
    # cleanest first: deepest dip, and largest zero-lag excess
    # deepest dip, but only among pairs with the counts to show one
    out['oversplit_dip'].sort(key=lambda r: (r['dip_ratio'], -r['flank_counts']))
    out['oversplit_nodip'].sort(key=lambda r: -r['dip_ratio'])
    return {k: v[:per_group] for k, v in out.items()}


# ---------------------------------------------------------------------------
# RANKING FOR READABILITY, measured in the space the panel actually draws.
#
# The first passes ranked on record fields (how many columns were quiet, how
# close the spike count sat to the median) and produced candidates that were
# statistically typical and visually mute -- Agrita rejected the offered
# examples in three of five rows. What a teaching row needs is that the
# evidence its rule read be VISIBLE, and that is a property of the plotted
# geometry, not of the record. So each kind is scored on the quantity a reader
# would look at:
#
#   clean       the unit sits clearly APART from its neighbour, is present
#               throughout, and has a strong peak-to-reference ratio
#   over-merge  the two imposed sub-populations separate in the BETWEEN-cluster
#               space as well, so the panel shows WHICH half sits on the
#               neighbour rather than one undifferentiated blob
#   intermittent  one long contiguous silence, not sparse firing everywhere

def isotropy(points):
    """How round a cloud is in the plotted 2-D: 0 = a line, 1 = circular.

    Some clusters collapse onto a single PC value and render as a vertical or
    horizontal line. They are not useful exemplars, and they SCORE WELL on
    centroid separation precisely because they have almost no spread in one
    direction -- the first clean ranking returned three of them in a row.
    """
    if points is None or len(points) < 20:
        return 0.0
    p = np.asarray(points)[:, :2]
    sd = p.std(0)
    return float(min(sd) / max(max(sd), 1e-12))


def _centroid_sep(a, b, min_isotropy=0.18):
    """Separation of two point sets in the plotted 2-D, in pooled SDs.

    Returns 0 when either cloud is degenerate, so a line-shaped cluster cannot
    win on separation it only has because it has no width.
    """
    if a is None or b is None or len(a) < 20 or len(b) < 20:
        return 0.0
    if isotropy(a) < min_isotropy or isotropy(b) < min_isotropy:
        return 0.0
    a2, b2 = np.asarray(a)[:, :2], np.asarray(b)[:, :2]
    d = np.linalg.norm(a2.mean(0) - b2.mean(0))
    spread = np.sqrt(0.5 * (a2.std(0).mean() ** 2 + b2.std(0).mean() ** 2))
    return float(d / max(spread, 1e-9))


def presence_contrast(times, duration_s=None, bin_s=30.0):
    """(longest silent fraction, active-vs-silent rate contrast).

    Intermittent presence should read as a unit that is THERE and then NOT,
    so the score rewards one long contiguous silence over a cluster that is
    merely sparse.
    """
    t = np.asarray(times, float)
    if t.size < 50:
        return 0.0, 0.0
    T = float(duration_s or t.max())
    if T <= 0:
        return 0.0, 0.0
    edges = np.arange(0, T + bin_s, bin_s)
    counts, _ = np.histogram(t, bins=edges)
    if counts.size < 4:
        return 0.0, 0.0
    nz = counts[counts > 0]
    # `counts[counts > 0] or [1]` raises on an array -- numpy has no truth
    # value for one. Every candidate was silently skipped by the caller's
    # except, so the whole row scored zero and looked like there were no good
    # examples rather than like a broken scorer.
    typical = float(np.median(nz)) if nz.size else 1.0
    active = counts > max(1.0, 0.1 * typical)
    best = run = 0
    for a in active:
        run = 0 if a else run + 1
        best = max(best, run)
    silent_frac = best / counts.size
    hi = counts[active].mean() if active.any() else 0.0
    lo = counts[~active].mean() if (~active).any() else 0.0
    return float(silent_frac), float(hi / max(lo, 0.5))
