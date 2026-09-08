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
                 chance against per-pair ground truth, so `mechanism` here says
                 which test spoke, never that the pair is settled.

WHAT IS NOT HERE YET. Division over-split -- two clusters that avoid each other at
short lag because one neuron cannot fire inside its own refractory period -- has
no detector in this tool, so no record is emitted for it. Measured corpus-wide:
6,758 such pairs, 0 flagged, 1,966 with both members passing. `mechanism` is an
open vocabulary so that channel can be added without changing this contract.

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
            out[key] = dict(
                a=x, b=y, channel=channel,
                decision_a=(ra or {}).get('decision'),
                decision_b=(rb or {}).get('decision'),
                outcome_a=(ra or {}).get('outcome'),
                outcome_b=(rb or {}).get('outcome'),
                # THE FIELD THE LEAK IS VISIBLE IN. Under every explanation of a
                # high-overlap pair -- one member is a copy, or both are one
                # artifact -- both members passing is wrong.
                both_pass=((ra or {}).get('decision') == 'PASS'
                           and (rb or {}).get('decision') == 'PASS'),
                mechanism=None, criterion_met=None, why=None,
                evidence={}, action='none')
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
            if p['overlap'] >= ev.get('overlap', -1.0):
                ev.update(overlap=p['overlap'], overlap_other=p['overlap_other'],
                          lag_ms=p['lag_ms'], over_lag_gate=p['over_lag_gate'])

    # a flagged pair is one where a member was declared a copy of the other
    for r in rows:
        a, partner = r['clu_id'], r.get('redundant_of')
        if partner is None:
            continue
        rec = _rec(a, int(partner), 're_detection')
        rec.update(mechanism='re_detection', criterion_met=True,
                   why='overlap and lag both over threshold',
                   action=f'drop_{"a" if rec["a"] == a else "b"}')

    for rec in out.values():
        if rec['channel'] != 're_detection' or rec['criterion_met']:
            continue
        ev = rec['evidence']
        rec['criterion_met'] = False
        rec['mechanism'] = 'shared_spikes_unadjudicated'
        # WHICH HALF OF THE CONJUNCTION FAILED is the useful part. `lag_below_
        # threshold` is the population that needs the duplicate-versus-shared-
        # artifact question answered before it can be given a verdict; the two
        # candidate answers need opposite actions, so this reports rather than
        # decides.
        rec['why'] = ('lag below LAG_MIN_MS: same spikes at ~zero lag, which is '
                      'either re-detection or one artifact seen twice -- not '
                      'adjudicated' if not ev.get('over_lag_gate') else
                      'overlap below this band\'s calibrated threshold')
        rec['action'] = 'review_pair' if rec['both_pass'] else 'none'

    # ---- channel 2: separability -------------------------------------------
    FROM_RULE = {'5a': ('separated_in_feature_space', 'none'),
                 '5b': ('coincidence_excess', 'review_pair'),
                 '5c': ('refractory_deficit', 'review_pair'),
                 '5d': ('evidence_incomplete', 'review_pair')}
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
            rec['evidence'].update(
                z=x.get('z'), n_min=x.get('n_min'), obs=x.get('obs'),
                lam=x.get('lam'), threshold=thr, n_accusers=k,
                over_threshold=(None if thr is None else x['z'] > thr))
            mech, act = FROM_RULE.get(str(rid), (None, 'none'))
            if mech:
                rec.update(mechanism=mech, criterion_met=(str(rid) != '5d'),
                           why=f'rule {rid} on the accused cluster', action=act)
            elif rec['mechanism'] is None:
                # THE ACCUSED CLUSTER NEVER REACHED THE NEIGHBOUR TEST. Accusers
                # are computed for any cluster mind_snag blames on a neighbour,
                # but an earlier rule can decide it first -- refractory or
                # contamination, say -- and then rule 5 never runs. The pair was
                # MEASURED and not ADJUDICATED, which is a different thing from
                # adjudicated-and-separate, so it is named rather than left null.
                rec.update(mechanism='accused_decided_earlier',
                           criterion_met=False,
                           why=f'the accused cluster was decided at rule {rid} '
                               f'before the neighbour test ran',
                           action='none')
        if not acc and r.get('dominant_neighbour') is not None and \
                str(rid or '').startswith('5'):
            rec = _rec(a, int(r['dominant_neighbour']), 'neighbour')
            mech, act = FROM_RULE.get(str(rid), ('not_separated', 'none'))
            rec.update(mechanism=mech, criterion_met=(str(rid) != '5d'),
                       why=f'rule {rid} on the accused cluster, no CCG statistic',
                       action=act)

    return [out[k] for k in sorted(out, key=lambda k: (k[0], k[1]))]


def summarise(pairs):
    """Counts a caller can print without re-deriving them."""
    from collections import Counter
    c = Counter(p['mechanism'] for p in pairs)
    return dict(
        n_pairs=len(pairs),
        by_mechanism=dict(c),
        n_both_pass=sum(1 for p in pairs if p['both_pass']),
        n_both_pass_unadjudicated=sum(
            1 for p in pairs
            if p['both_pass'] and p['mechanism'] == 'shared_spikes_unadjudicated'),
        n_action_review=sum(1 for p in pairs if p['action'] == 'review_pair'),
        n_action_drop=sum(1 for p in pairs if p['action'].startswith('drop')))
