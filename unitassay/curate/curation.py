#!/usr/bin/env python3
"""curate/curation.py — the toolbox entry point. ONE wrapper, one pipeline.

    from unitassay.curate.curation import curate_session
    rows = curate_session(ks4_dir, isolation_dir='isolation')

WHY THIS EXISTS
---------------
`decision_flow.curate()` is a pure function over a per-cluster dict, and it has
always had exactly one implementation. **Assembling that dict did not.** The
sequence

    load KS4 -> mind_snag SortData -> channel_fields -> overmerge_verdict
             -> ISI-VR -> re-detection (two passes) -> curate -> classify
             -> findings/error_class

was written out independently in `m260820_whole_tool_performance.py`,
`m260820_ks4_vs_tool.py`, `mearec_synthetic/score_tag_mapping.py`,
`mearec_synthetic/hybrid_compare_tools.py`, `mearec_synthetic/rule5_yield_check.py`
and `m260828_rule1_three_arm.py`. Six copies of the pipeline feeding one copy of
the rule.

That is not a tidiness complaint. It is where a real bug lived: every one of
those copies paraphrased mind_snag's ISI-VR as

    passes_isi = not (len(viol) >= 5) or burst_like

which keeps the small-n FLOOR and silently drops the Hill ratio behind it,
turning a rate test into an absolute count. It wrongly failed 223 of 4,016
synthetic clusters — none below 1,000 spikes, which is why it survived — and on
Buccino's hybrid corpus (54k–85k spikes per unit) it failed 8 of 9 ground-truth
units, making `curate()` look like an 11% -sensitivity tool. The MEArec terminal
found it from the outside; neither of us could see it from inside the corpus.

**The rule was never wrong. The six pipelines were.** So the pipeline gets one
implementation too, and callers get a function rather than a recipe.

WHAT IT DOES NOT DO, deliberately
---------------------------------
* No ground-truth scoring, no figures, no I/O to `data/results`. Those belong to
  the analysis scripts; this returns verdicts and nothing else.
* No patch loading. `curate/records.py` owns the 50-session record set and its
  corrections; this is for curating a sort from disk. The two meet at
  `curate()`, which both call.
* No burst-override reimplementation — it needs PC features and an LDA test, so
  it is computed here where those are already loaded, and passed to
  `curate.isi_vr.passes_isi_vr` as the `or is_burst` term its docstring specifies.

ORDER MATTERS AND IS NOT NEGOTIABLE
-----------------------------------
Rule R needs `redundant_of`, which needs to know which clusters rule 1 claimed,
which needs a `curate()` call. So curation runs TWICE: pass 1 without
redundancy to establish rule 1, pass 2 with it. Rule 1 precedes rule R in the
precedence, so this is well-founded rather than circular. Getting this wrong is
how rule R sat silently inactive in every score the MEArec terminal published —
`redundant_of` was never populated, so a rule that could not fire was
indistinguishable from a rule that fired and found nothing.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""
import json
from pathlib import Path

import warnings

import numpy as np

from unitassay.curate import redetection as RD
from unitassay.curate.bimodality import overmerge_verdict
from unitassay.curate.constants import MIN_GROUP_SPIKES
from unitassay.curate.decision_flow import curate
from unitassay.curate.isi_vr import passes_isi_vr, REFRACTORY_MS
from unitassay.curate.taxonomy import curate_and_classify
from unitassay.curate.bimodality import dip_lratio_between
from unitassay.curate.ccg import accuser_stats
from unitassay.curate.constants import DIP_PVAL_THRESH, L_RATIO_THRESH
# The between-cluster LDL uses the SAME per-group floor as the within-cluster
# one -- 120 spikes. m260729_between_cluster_real_check named it
# MIN_GROUP_SPIKES_LDL locally; it is MIN_GROUP_SPIKES here, imported below with
# the rest, so there is one value rather than two that can drift.
from unitassay.curate.sortdata import (aligned_features, flagged_neighbours,
                            sortdata_path, channel_fields,
                            process_sortdata_all_frames)

MIN_SPIKES = 50
N_PC = 3
BURST_SUB_MS = 0.5        # violations inside this sub-window look burst-like
BURST_FRAC = 0.80         # ...at this concentration or above
SHAPE_SIM = 0.4           # clus_channel_info's noise-reference threshold


def _sample_rate(ks4_dir, default=30000.0):
    """KS4 writes it to params.py. MEArec sorts are 32 kHz, real ones 30 kHz,
    and getting this wrong rescales every ISI in the pipeline."""
    p = Path(ks4_dir) / 'params.py'
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip().startswith('sample_rate'):
                return float(ln.split('=')[1].strip())
    return default


def cluster_inputs(spike_times_ms, pc_feats, channel_fields_row, duration_sec):
    """curate()'s per-cluster dict, for ONE cluster. No I/O.

    Split out from `curate_session` so a caller holding its own loaded arrays
    (a synthetic harness, a hybrid case) can reuse the exact input contract
    without re-deriving it — which is what every one of the six copies was
    doing by hand.
    """
    t = np.sort(np.asarray(spike_times_ms, float))
    isis = np.diff(t)
    viol = isis[isis < REFRACTORY_MS]
    # Burst override: violations piled into the first half of the refractory
    # window look like an intrinsic burster rather than contamination. Timing
    # alone cannot separate that from a real over-merge, which is why mind_snag
    # gates it on an LDA bimodality test; that gate lives in auto_curate and is
    # not reproduced here, so this is the timing half only.
    is_burst = bool(len(viol) and np.mean(viol < BURST_SUB_MS) >= BURST_FRAC)
    passes_isi = passes_isi_vr(t) or is_burst

    vv = overmerge_verdict(pc_feats, min_spikes=MIN_SPIKES,
                           min_group_spikes=MIN_GROUP_SPIKES,
                           duration_sec=duration_sec)
    n_viol = int(len(viol))
    n = len(t)
    ch = channel_fields_row

    # The two rule-1 three-way inputs. Both are RAW and unthresholded on
    # purpose: `passes_isi` above is the yes/no mind_snag derives from the same
    # spikes and discards, and the continuous values still discriminate among
    # clusters that all passed it.
    isi_rate = (n_viol / (n - 1)) if n > 1 else None
    split_time_auc = _split_time_auc(vv.get('proj'), t)

    return dict(
        ldl_verdict=vv['verdict'],
        screen_pass=bool(ch['passes_lratio'] and passes_isi),
        passes_isi_independent=bool(passes_isi),
        dominant_source=ch['dominant_source'],
        noise_ch_flagged=ch.get('noise_ch_flagged'),
        neighbor_never_fails=ch['neighbor_never_fails'],
        # frame coverage -- what separates "absent" from "contaminated" when a
        # cluster fails mind_snag with every attribution channel clean
        absent_frame_frac=ch.get('absent_frame_frac'),
        clean_frac_when_present=ch.get('clean_frac_when_present'),
        between_cluster_confirmed_separated=None,
        between_all_neighbours_separated=None,
        dip_basis=vv['basis'],
        n_spikes=int(pc_feats.shape[0]),
        is_burst=is_burst,
        isi_rate=isi_rate,
        split_time_auc=split_time_auc,
        # diagnostics, not consumed by curate()
        _n_viol=n_viol, _dip_pval=vv.get('dip_pval'), _l_ratio=vv.get('l_ratio'),
    )


def _split_time_auc(proj, t_ms):
    """How time-ordered is the LDL's own split? 0.5 interleaved, 1.0 early-late.

    Rule 1's drift arm needs this. Returns None when no split was formed or a
    side is too small — and None is NOT "not drift": curate() holds such a
    cluster in the drift arm rather than releasing it.
    """
    if proj is None:
        return None
    p = np.asarray(proj, float)
    if len(p) != len(t_ms):
        return None
    fin = np.isfinite(p)
    if fin.sum() < 20:
        return None
    p, t = p[fin], np.asarray(t_ms, float)[fin]
    g = p > np.median(p)
    if not (5 < g.sum() < len(g) - 5):
        return None
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(t[g], t[~g])
    return float(abs(u / (g.sum() * (~g).sum()) - 0.5) + 0.5)


def curate_session(ks4_dir, isolation_dir='isolation', cluster_ids=None,
                   duration_sec=None, with_redetection=True, verbose=False,
                   mindsnag_subdir=None):
    """Curate one sorted session. Returns one dict per cluster.

    ks4_dir           KS4 output directory (spike_times/clusters, pc_features,
                      templates, channel_positions, params.py)
    isolation_dir     where the screen layer's SortData.h5 files are. An
                      ABSOLUTE PATH is the safe form and is what the CLI passes;
                      a bare NAME asks this function to search beside and inside
                      ks4_dir, and it refuses when both exist rather than
                      preferring one.
    cluster_ids       restrict to these (0-indexed KS4 ids); default all with
                      a SortData file
    duration_sec      recording length. Defaults to the last spike time, which
                      is what the satellite-rate test needs — omitting it once
                      disabled that test entirely on real data.
    mindsnag_subdir   DEPRECATED alias for isolation_dir, kept because callers
                      outside this package still pass it.

    THE DEFAULT CHANGED WITH THE NAME, from 'mindsnag' to 'isolation'. The
    layer is called `screen` and its CLI writes to `isolation/`, so the old
    default named a directory this package never creates. A caller relying on
    the old default was depending on a lab layout, not on the tool.
    """
    if mindsnag_subdir is not None:
        # NOT silently accepted: the two names would otherwise disagree about
        # the default, and a caller passing the old one would get 'isolation'
        # searched for while believing it had asked for something else.
        warnings.warn('curate_session(mindsnag_subdir=) is deprecated; use '
                      'isolation_dir=. The screen layer replaced mind_snag '
                      'and the default is now "isolation".',
                      DeprecationWarning, stacklevel=2)
        isolation_dir = mindsnag_subdir
    # clus_channel_info is mind_snag's, so it stays a deferred import -- the
    # dev tree is not always on sys.path and only this function needs it.
    # NO PIN NEEDED HERE. In the lab tree this import had to force a specific
    # mind_snag onto sys.path, because three diverged copies existed and the one
    # `import mind_snag` found had a different clus_channel_info signature and a
    # different noise-reference rule. Inside this package there is exactly one
    # copy -- unitassay.screen -- so the ambiguity that made the pin necessary
    # does not exist.
    from unitassay.screen.utils.channel_info import clus_channel_info

    ks4 = Path(ks4_dir)
    # TWO LAYOUTS EXIST AND sortdata_path TAKES THE PARENT OF THE SortData DIR.
    #   MEArec  recordings_X/{ks4_output, mindsnag_dev_w100}   -> ks4.parent
    #   real    group_CatGT_KS4/{spike_times.npy, mindsnag/}   -> ks4 itself
    # Detected rather than assumed: hardcoding either one silently curates ZERO
    # clusters on the other, because every sortdata_path() misses and the loop
    # `continue`s. Raise if neither is present -- an empty result must not be
    # mistakable for "this session has no clusters".
    #   AN ABSOLUTE PATH IS ALSO ACCEPTED, which is how the CLI uses this: it
    # writes isolation under --out, which need not be anywhere near --ks-dir.
    # Without this branch the two disagreed about where the files were and the
    # run died after the screen stage had already done all its work.
    _sub = Path(isolation_dir)
    if _sub.is_absolute():
        if not _sub.is_dir():
            raise FileNotFoundError(f'no isolation directory at {_sub}')
        root, isolation_dir = _sub.parent, _sub.name
    else:
        # A BARE NAME ASKS THIS FUNCTION TO SEARCH, and searching is how a whole
        # 147-session run was curated from stale files on 2026-09-04: a
        # pre-existing directory of the same name sat inside every sorter
        # directory, and the search found it instead of the isolation that run
        # had just written. It did not fail; it returned a complete, plausible
        # set of verdicts from the wrong data. So when both candidates exist,
        # refuse rather than prefer.
        here, up = ks4 / isolation_dir, ks4.parent / isolation_dir
        if here.is_dir() and up.is_dir():
            raise FileNotFoundError(
                f'{isolation_dir!r} exists BOTH inside ({here}) and beside '
                f'({up}) the sorter directory, so which one is meant is '
                f'ambiguous. Pass an absolute path to say which.')
        if not (here.is_dir() or up.is_dir()):
            raise FileNotFoundError(
                f'no {isolation_dir!r} directory beside or inside {ks4}; '
                'curating would silently return nothing')
        root = ks4 if here.is_dir() else ks4.parent
    fs = _sample_rate(ks4)
    st_ms = np.load(ks4 / 'spike_times.npy').ravel().astype(float) / fs * 1000.0
    sc = np.load(ks4 / 'spike_clusters.npy').ravel()
    pc = np.load(ks4 / 'pc_features.npy', mmap_mode='r')
    npc = min(N_PC, pc.shape[1])
    if duration_sec is None:
        duration_sec = float(st_ms.max()) / 1000.0 if len(st_ms) else 0.0
    try:
        _, _, flagged = clus_channel_info(str(ks4), shape_sim_threshold=SHAPE_SIM)
    except Exception:
        flagged = None

    ids = sorted(set(sc.tolist())) if cluster_ids is None else list(cluster_ids)
    base, meta = {}, {}
    for clu in ids:
        f = sortdata_path(root, clu, isolation_dir)
        if not f.exists():
            continue
        ch = channel_fields(process_sortdata_all_frames(f))
        if ch is None:
            continue
        idx = np.where(sc == clu)[0]
        if len(idx) == 0:
            continue
        X = np.asarray(pc[idx, :npc, :]).reshape(len(idx), -1)
        ok = np.all(np.isfinite(X), axis=1)
        X = X[ok]
        if X.shape[0] == 0:
            continue
        if flagged is not None and clu < len(flagged):
            ch = dict(ch, noise_ch_flagged=bool(flagged[clu]))
        d = cluster_inputs(st_ms[idx][ok], X, ch, duration_sec)
        meta[clu] = {k: d.pop(k) for k in list(d) if k.startswith('_')}
        base[clu] = d

    # --- re-detection, two passes (see module docstring) ---------------------
    verdicts, scanned, overlap_pairs = {}, set(), {}
    if with_redetection and base:
        try:
            scan = RD.scan_session(np.load(ks4 / 'templates.npy'), st_ms, sc,
                                   np.load(ks4 / 'channel_positions.npy'))
            thr, _, _ = RD.band_thresholds(scan)
            rule1 = {c: str(curate(d)['rule']).startswith('1')
                     for c, d in base.items()}
            verdicts = RD.redundancy_verdicts(scan, thr,
                                              lambda x: bool(rule1.get(x)))
            scanned = {sv['clu'] for sv in scan}
            # KEEP EVERY HIGH-OVERLAP PAIR, not only the ones the criterion
            # flagged. `is_redetection_pair` is a CONJUNCTION -- overlap AND a
            # non-zero lag -- and on real data 98.6% of pairs over the overlap
            # floor fail the lag half (23,956 measured corpus-wide, 331 over the
            # gate). Discarding them here is why a pair that shares most of one
            # cluster's spikes can leave no trace in the output at all: 275 such
            # pairs have BOTH members passing.
            #
            # Retaining them asserts nothing. The lag requirement is what makes
            # the criterion mean "re-detection" rather than "these two fire
            # together", and it stays exactly as it was -- these rows carry the
            # measurement and say the criterion was not met, which is a
            # different statement from silence.
            for sv in scan:
                keep = [dict(other=int(b), overlap=float(oab),
                             overlap_other=float(oba), lag_ms=float(lag),
                             over_lag_gate=bool(abs(lag) > RD.LAG_MIN_MS))
                        for b, oab, oba, _nb, _amp, lag in sv['pairs']
                        if max(oab, oba) >= RD.OVERLAP_FLOOR]
                if keep:
                    overlap_pairs[sv['clu']] = keep
        except Exception as e:
            # LOUD, not silent. A failed scan means rule R cannot fire, which is
            # exactly the "absent is not negative" failure this module documents.
            raise RuntimeError(
                f'{ks4}: re-detection scan failed ({type(e).__name__}: {e}); '
                'refusing to curate with rule R silently inactive') from e

    import h5py

    # ---- rule 5's inputs, built HERE (task A11, 2026-09-04) -----------------
    # These used to be computed only in analysis/m260902_full_synthetic_curate.py,
    # so a bare curate_session() on a fresh session RAISED at rule 5's retrieval
    # -- correctly, since a retrieval that passes because its check was absent is
    # an untested default, but it meant the toolbox could not curate a session on
    # its own. Everything below reads files curate_session already opened.
    def _between(a, b):
        """Do two clusters separate in the feature space they SHARE?"""
        fa, fb, nsh = aligned_features(pc, pfi, int(a), int(b), sc, n_pc=npc)
        if fa is None or nsh < 2:
            return None
        try:
            res = dip_lratio_between(fa, fb, min_group_spikes=MIN_GROUP_SPIKES)
        except Exception:
            return None
        dp, lr = res.get('dip_pval'), res.get('l_ratio')
        if dp is None or lr is None or not (np.isfinite(dp) and np.isfinite(lr)):
            return None
        if dp < DIP_PVAL_THRESH and lr < L_RATIO_THRESH:
            return True
        if dp >= DIP_PVAL_THRESH and lr >= L_RATIO_THRESH:
            return False
        return None                      # the two statistics conflict

    # RAW SAMPLES, not milliseconds. compute_ccg's docstring says "spike times
    # in RAW SAMPLES (divided by fs internally)", so handing it st_ms silently
    # rescales every lag by fs/1000 -- it changed 365 accuser sets and 31
    # decisions before this was caught by diffing against the pre-move run.
    st_samp = st_ms * fs / 1000.0
    times = {int(c): np.sort(st_samp[sc == c]) for c in np.unique(sc)}
    pfi = np.load(ks4 / 'pc_feature_ind.npy')
    for clu, d in base.items():
        d.update(between_cluster_confirmed_separated=None,
                 between_all_neighbours_separated=None, ccg_accusers=[])
        # only clusters mind_snag blames on a NEIGHBOUR reach rule 5's retrieval
        if d.get('dominant_source') != 'other_cluster' or d.get('screen_pass'):
            continue
        f = sortdata_path(root, clu, isolation_dir)
        if not f.exists():
            continue
        fl, _all = flagged_neighbours(f)
        sd = process_sortdata_all_frames(f)
        if sd is None or not sd['lr_others']:
            continue
        with h5py.File(f, 'r') as h:
            other = ([int(x) - 1 for x in h['frame_0000']['other_clu'][:]]
                     if 'frame_0000' in h and 'other_clu' in h['frame_0000'] else [])
        if not other:
            continue
        per = [np.nanmean(x) if not np.all(np.isnan(x)) else np.nan
               for x in sd['lr_others']]
        if np.all(np.isnan(per)):
            continue
        dom = other[int(np.nanargmax(per))]
        d['dominant_neighbour'] = int(dom)
        d['between_cluster_confirmed_separated'] = _between(clu, dom)
        if d['between_cluster_confirmed_separated'] is not True:
            continue
        # EVERY accuser, not the loudest (#14). Keying on the dominant neighbour
        # alone left 31% of this branch with accusers the gate never saw.
        acc = [nb['clu_id'] for nb in (fl or []) if nb['clu_id'] != clu] or [dom]
        seps = [_between(clu, x) for x in acc]
        d['between_all_neighbours_separated'] = (
            True if all(s is True for s in seps)
            else (False if any(s is False for s in seps) else None))
        got = []
        for nb in acc:
            s = accuser_stats(times.get(int(clu)), times.get(int(nb)), fs=fs)
            if s is not None:            # None = NOT MEASURABLE, not clean
                got.append(dict(clu=int(nb), **s))
        d['ccg_accusers'] = got

    rows = []
    for clu, d in base.items():
        partner = verdicts.get(clu) if clu in scanned else None
        row = dict(d, redundant_of=None if partner is None else int(partner),
                   overlap_pairs=overlap_pairs.get(clu, []))
        # `meta` is the `_`-prefixed diagnostics only, and curate() reads none of
        # them, so merging it before the verdict cannot change a decision.
        row.update(meta.get(clu, {}))
        row['clu_id'] = int(clu)
        rows.append(row)

    # DECIDE, THEN DESCRIBE -- never one without the other. This module's own
    # docstring has claimed `-> curate -> classify` since it was written while
    # the code returned straight after curate(), leaving every caller to classify
    # for itself. That is the gap that let `intermittent_presence` fire on 41
    # real clusters and 0 synthetic ones for eleven days: the decision layer
    # looked complete, and nothing ever asked the description layer whether it
    # could name what happened. Agrita's rule, 2026-09-02.
    curate_and_classify(rows)

    # findings/outcome are attached by curate_and_classify above -- all three
    # layers together, because findings read enrichments and must be recomputed
    # whenever the record changes.
    if verbose:
        print(f'{ks4}: {len(rows)} clusters curated '
              f'({sum(r["decision"] == "PASS" for r in rows)} pass)')
    return rows
