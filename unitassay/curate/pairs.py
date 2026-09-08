"""curate/pairs.py — the pairwise view of a curated session.

WHY THIS EXISTS. `curate()` decides per CLUSTER, but duplication and over-splitting
are properties of a PAIR: a fragment is a fragment *of* something. Measured on 147
real sessions, every cluster labelled `duplicate` names its partner (382 of 382)
while only 20.7% of those labelled `oversplit` do -- the other 79.3% are a pairwise
claim made from a within-cluster test that never looks at another cluster. That
mismatch is why pairwise errors are under-reported: 23,956 pairs share at least
half of one member's spikes in the same millisecond bin, and 275 of them have BOTH
members passing.

WHAT THIS DOES AND DOES NOT DO. It emits, one record per pair, the evidence the
tool already computed and what it concluded. It runs no new statistic, changes no
verdict, and assigns no cluster a new label. Every field is derived from rows
`curate_session` already returns.

THE TWO CHANNELS, and they answer different questions:

  re_detection   do these two clusters contain the SAME SPIKES?
                 overlap within +-0.5 ms of the pair's modal lag, plus that lag.
                 The tool's criterion is a CONJUNCTION -- overlap AND |lag| >
                 LAG_MIN_MS -- and the lag half is what makes it mean
                 "re-detection" rather than "these two fire together". Pairs over
                 the overlap floor that fail the lag half are reported here with
                 `criterion_met=False` and the reason, rather than omitted. On
                 real data that is 98.6% of them.

  neighbour      are these two clusters SEPARABLE? Feature space adjudicates
                 (rule 5a) and the time domain gates it (5b coincidence excess,
                 5c refractory deficit). The feature-space arm is measured at
                 chance against per-pair ground truth, so `decided_by` here says
                 which test spoke, never that the pair is settled.

NO NEW VOCABULARY. A pair record says which QUESTION was asked (`channel`),
whether the criterion fired (`criterion_met`), which rule adjudicated it
(`decided_by`, the same `rule_id` the README's table is indexed by) and why in
words. It deliberately does NOT add a `mechanism` field: rows already have one,
with a different 9-value vocabulary, and two meanings for one word in one tool is
how a reader ends up trusting the wrong column. This tool already carries eight
per-cluster vocabularies; it does not need a ninth.

WHAT IS NOT HERE YET. Division over-split -- two clusters that avoid each other at
short lag because one neuron cannot fire inside its own refractory period -- has
no detector in this tool, so no record is emitted for it. Measured corpus-wide:
6,758 such pairs, 0 flagged, 1,966 with both members passing. Adding it means a
new `decided_by` rule, not a new field.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
from unitassay.curate.constants import ccg_z_threshold


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def pair_findings(rows):
    """One record per pair the tool measured. Pure function over curated rows.

    Returns a list of dicts sorted by (a, b), each carrying both members'
    verdicts, the evidence, what the tool concluded, and a suggested action.
    """
    by = {r['clu_id']: r for r in rows}
    out = {}

    def _rec(a, b, channel):
        key = (_canon(a, b), channel)
        if key not in out:
            (x, y) = key[0]
            ra, rb = by.get(x), by.get(y)
            # EIGHT FIELDS, and each one is here because dropping it loses
            # something. `decision_a/b` and `outcome_a/b` were cut: they copy
            # curated.json, which the reader already has and can join on clu_id.
            # What survives of them is `both_pass`, the one derived fact worth
            # filtering on -- under every explanation of a high-overlap pair
            # (one member is a copy, or both are one artifact) both members
            # passing is wrong. No `mechanism` field: rows already have one with
            # a different vocabulary, and this tool has eight per-cluster word
            # lists already. It does not need a ninth.
            out[key] = dict(
                a=x, b=y, channel=channel,
                both_pass=((ra or {}).get('decision') == 'PASS'
                           and (rb or {}).get('decision') == 'PASS'),
                criterion_met=None, evidence={}, why=None, action='none')
        return out[key]

    # ---- channel 1: shared spikes ------------------------------------------
    for r in rows:
        a = r['clu_id']
        for p in (r.get('overlap_pairs') or []):
            b = int(p['other'])
            rec = _rec(a, b, 're_detection')
            ev = rec['evidence']
            # the scan lists a pair from both sides; keep the larger containment,
            # which is the direction the criterion is applied in
            # THE LARGER CONTAINMENT, which is the number the criterion is
            # applied to (`is_redetection_pair` is tried in both directions).
            # Reporting both directions invited the reader to work out which
            # cluster is inside which, and for an unadjudicated pair that is not
            # actionable anyway -- for an adjudicated one `action` already says
            # which member to drop.
            o = max(p['overlap'], p['overlap_other'])
            if o >= ev.get('overlap', -1.0):
                ev.update(overlap=o, lag_ms=p['lag_ms'])

    # a flagged pair is one where a member was declared a copy of the other
    for r in rows:
        a, partner = r['clu_id'], r.get('redundant_of')
        if partner is None:
            continue
        rec = _rec(a, int(partner), 're_detection')
        rec.update(criterion_met=True,
                   why='overlap and lag both over threshold',
                   action=f'drop_{"a" if rec["a"] == a else "b"}')

    for rec in out.values():
        if rec['channel'] != 're_detection' or rec['criterion_met']:
            continue
        ev = rec['evidence']
        rec['criterion_met'] = False
        # WHICH HALF OF THE CONJUNCTION FAILED is the useful part. `lag_below_
        # threshold` is the population that needs the duplicate-versus-shared-
        # artifact question answered before it can be given a verdict; the two
        # candidate answers need opposite actions, so this reports rather than
        # decides.
        rec['why'] = ('same spikes at ~zero lag: re-detection or one artifact '
                      'seen twice, not adjudicated'
                      if abs(ev.get('lag_ms', 0.0)) <= 0.2 else
                      'overlap below this band\'s calibrated threshold')
        rec['action'] = 'review_pair' if rec['both_pass'] else 'none'

    # ---- channel 2: separability -------------------------------------------
    # the action each neighbour rule implies; the rule_id itself is the label
    ACTION = {'5a': 'none', '5b': 'review_pair',
              '5c': 'review_pair', '5d': 'review_pair'}
    for r in rows:
        a, rid = r['clu_id'], r.get('rule_id')
        acc = r.get('ccg_accusers') or []
        usable = [x for x in acc if x.get('z') is not None
                  and x.get('n_min') is not None]
        k = len(usable) or 1
        seen = set()
        for x in acc:
            b = x.get('clu')
            if b is None or b in seen:
                continue
            seen.add(int(b))
            rec = _rec(a, int(b), 'neighbour')
            thr = (ccg_z_threshold(x['n_min'], k)
                   if x.get('z') is not None and x.get('n_min') is not None
                   else None)
            # z against its own threshold is the whole statement; obs/lam are
            # the Poisson internals behind z and `over_threshold` is z > thr.
            rec['evidence'].update(z=x.get('z'), threshold=thr,
                                   n_spikes_min=x.get('n_min'))
            if str(rid) in ACTION:
                rec.update(criterion_met=(str(rid) != '5d'),
                           why=f'rule {rid} adjudicated the accused cluster',
                           action=ACTION[str(rid)])
            elif rec['criterion_met'] is None:
                # THE ACCUSED CLUSTER NEVER REACHED THE NEIGHBOUR TEST. Accusers
                # are computed for any cluster mind_snag blames on a neighbour,
                # but an earlier rule can decide it first -- refractory or
                # contamination, say -- and then rule 5 never runs. The pair was
                # MEASURED and not ADJUDICATED, which is a different thing from
                # adjudicated-and-separate, so it is named rather than left null.
                rec.update(criterion_met=False,
                           why=f'the accused cluster was decided at rule {rid} '
                               f'before the neighbour test ran',
                           action='none')
        if not acc and r.get('dominant_neighbour') is not None and \
                str(rid or '').startswith('5'):
            rec = _rec(a, int(r['dominant_neighbour']), 'neighbour')
            rec.update(criterion_met=(str(rid) != '5d'),
                       why=f'rule {rid} adjudicated the accused cluster; no CCG '
                           f'statistic was available',
                       action=ACTION.get(str(rid), 'none'))

    return [out[k] for k in sorted(out, key=lambda k: (k[0], k[1]))]


def summarise(pairs):
    """Counts a caller can print without re-deriving them."""
    from collections import Counter
    # four numbers. Anything else a caller wants is one filter over `pairs`.
    return dict(
        n_pairs=len(pairs),
        by_channel=dict(Counter(p['channel'] for p in pairs)),
        n_not_adjudicated=sum(1 for p in pairs if not p['criterion_met']),
        n_both_pass=sum(1 for p in pairs if p['both_pass']))
