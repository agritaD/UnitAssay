"""Figure 8 rows: one cluster drawn across the six columns of evidence.

THE RECORD DECIDES, THIS MODULE ONLY DRAWS. Every statistic shown beside a
panel is read from the stored curate() record. Recomputing a verdict here is
how the earlier rule figures ended up captioned with one test's answer and
drawn with another's, twice. Geometry (the LDA axis, the group labels) is
recomputed because it is not stored; verdicts never are.

COLOUR ENCODES SCOPE, because scope is the thing the rules are organised by:
decision_flow's own note says a stage is a scope, not an outcome -- a
within-cluster test is blind to other clusters, and only a between-cluster
test can say "other cluster". So the grammar is:

    blue    this cluster, in every panel
    orange  the part the dip peeled off it       WITHIN-cluster columns only
    red     another cluster on the same channel  BETWEEN-cluster columns
    pink    a second such cluster
    grey    the reference channel layer 1 measured against

Columns, left to right -- the two within-cluster columns adjacent, then the
two between-cluster ones, so the grammar reads in blocks:

    waveform | ISI | KS4 feature space | 1-D projection | amplitude | CCG
      all       r2         r4a/5a            r3a            r3b      r1/5b/5c

Palette validated all-pairs for CVD (worst pair dE 15.6 deutan). The orange
and pink sit under 3:1 against white, so every panel that uses them carries a
visible label -- which the validator requires as relief, and which the panels
need anyway to say WHICH cluster is which.
"""
import math
from pathlib import Path

import numpy as np

from unitassay.curate.bimodality import overmerge_verdict
from unitassay.curate.ccg import CCG_BIN_MS, CCG_WINDOW_MS, compute_ccg
from unitassay.curate.constants import ABSENT_FRAME_FRAC_THRESH, DIP_PVAL_THRESH
from unitassay.curate.sortdata import aligned_features
from unitassay.screen.io.ks_loader import load_ks_dir

N_PC = 3
FS = 30000.0

C_UNIT = '#3953A4'
C_SPLIT = '#F58430'
C_OTHER = ('#BF1E2D', '#E386B7')
C_REF = '#8A8F98'
C_FIRE = '#D55E00'
C_QUIET = '#5B6470'
C_HIST = '#6E7681'


def _isfinite(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def _flat_feats(sp, clu_id):
    idx = np.where(sp.clu == clu_id)[0]
    if sp.pc_feat is None or not idx.size:
        return None
    npc = min(N_PC, sp.pc_feat.shape[1])
    flat = sp.pc_feat[idx, :npc, :].reshape(idx.size, -1)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    return flat if flat.size else None


def load_row(session_dir, ks_dir, clu_id, ks_cache=None, max_others=2):
    """Everything the six columns need, including the same-channel neighbours.

    The neighbours come from the screen's own SortData -- they ARE layer 1's
    "other" clusters, the ones its L-ratio was measured against -- so drawing
    them is what connects this figure to the reference/neighbour idea Figure 7
    introduces. The CCG partner the RULE used is added even when it is not one
    of them, because the CCG column has to show the rule's evidence.
    """
    ks_cache = {} if ks_cache is None else ks_cache
    if ks_dir not in ks_cache:
        ks_cache[ks_dir] = load_ks_dir(ks_dir, exclude_noise=False)
    sp = ks_cache[ks_dir]

    idx = np.where(sp.clu == clu_id)[0]
    out = {'clu_id': clu_id, 'n_spikes': int(idx.size),
           'times': sp.st[idx], 'amps': sp.temp_scaling_amps[idx],
           'feats': _flat_feats(sp, clu_id), 'wf': None, 'noise_wf': None,
           'screen_unit': None, 'screen_noise': None, 'others': []}

    p = Path(session_dir) / 'isolation' / f'{clu_id + 1}.SortData.h5'
    neigh = []
    if p.exists():
        import h5py
        with h5py.File(p, 'r') as h:
            frames = sorted(h.keys())
            cw = [h[f]['clu_wf'][:] for f in frames if 'clu_wf' in h[f]]
            nw = [h[f]['noise_wf'][:] for f in frames if 'noise_wf' in h[f]]
            u = [h[f]['unit'][:] for f in frames if 'unit' in h[f]]
            nz = [h[f]['noise'][:] for f in frames if 'noise' in h[f]]
            wf_by_clu = {}
            for f in frames:
                # Not every frame carries both: a frame can name a neighbour
                # without storing its waveform. Collect ids regardless, and
                # waveforms only where they exist.
                if 'other_clu' not in h[f]:
                    continue
                ids = np.atleast_1d(h[f]['other_clu'][:]).astype(int)
                wfs = (np.atleast_2d(h[f]['other_clu_wf'][:])
                       if 'other_clu_wf' in h[f] else None)
                for j, cid in enumerate(ids):
                    ent = wf_by_clu.setdefault(int(cid), [])
                    if wfs is not None and j < len(wfs):
                        ent.append(wfs[j])
        if cw:
            out['wf'] = np.mean(np.vstack(cw), axis=0)
        if nw:
            out['noise_wf'] = np.mean(np.vstack(nw), axis=0)
        if u:
            out['screen_unit'] = np.vstack(u)
        if nz:
            out['screen_noise'] = np.vstack(nz)
        neigh = [(c, (np.mean(np.vstack(v), axis=0) if v else None))
                 for c, v in wf_by_clu.items()]

    out['_sp'] = sp
    for cid, wf in neigh[:max_others]:
        oidx = np.where(sp.clu == cid)[0]
        out['others'].append({
            'clu_id': cid, 'wf': wf, 'role': 'same channel',
            'times': sp.st[oidx], 'amps': sp.temp_scaling_amps[oidx],
            'feats': _flat_feats(sp, cid)})
    return out


def align_primary(data):
    """Align the unit with its primary neighbour on the channels they share.

    Called after add_rule_partner() so the pair drawn is the one the rule
    used, falling back to the screen's same-channel neighbour.
    """
    sp = data.get('_sp')
    others = data.get('others') or []
    if sp is None or sp.pc_feat is None or not others:
        return data
    prt = next((o for o in others if o.get('is_partner')), others[0])
    fa, fb, n_shared = aligned_features(sp.pc_feat, sp.pc_feat_ind,
                                        data['clu_id'], prt['clu_id'], sp.clu)
    if fa is not None and len(fa) and fb is not None and len(fb):
        data['fs_aligned'] = (fa, fb, prt['clu_id'], n_shared)
    return data


def add_rule_partner(data, rec_row, sp_cache_entry):
    """Ensure the partner the RULE used is present, flagged with why."""
    have = {o['clu_id'] for o in data['others']}
    pid, why = None, None
    if rec_row.get('redundant_of') is not None:
        pid, why = int(rec_row['redundant_of']), 'counted under'
    elif rec_row.get('overlap_pairs'):
        p = rec_row['overlap_pairs'][0]
        pid, why = int(p['clu'] if isinstance(p, dict) else p), 'overlapping'
    else:
        acc = [a for a in (rec_row.get('ccg_accusers') or [])
               if isinstance(a, dict) and a.get('z') is not None]
        if acc:
            pid, why = int(max(acc, key=lambda a: a['z'])['clu']), 'accuser'
    if pid is None:
        return data
    for o in data['others']:
        if o['clu_id'] == pid:
            o['role'] = why
            o['is_partner'] = True
            return data
    sp = sp_cache_entry
    oidx = np.where(sp.clu == pid)[0]
    if oidx.size:
        data['others'].insert(0, {
            'clu_id': pid, 'wf': None, 'role': why, 'is_partner': True,
            'times': sp.st[oidx], 'amps': sp.temp_scaling_amps[oidx],
            'feats': _flat_feats(sp, pid)})
    return data


def draw_row(axes, rec_row, data, show_xlabels=False):
    """Draw one cluster across six axes: wf, isi, feature space, projection,
    amplitude, ccg."""
    a_wf, a_isi, a_fs, a_prj, a_amp, a_ccg = axes
    fired = rec_row.get('_fires')
    others = data.get('others') or []
    ocol = {o['clu_id']: C_OTHER[i % len(C_OTHER)] for i, o in enumerate(others)}

    def frame(ax, col):
        for s in ax.spines.values():
            s.set_linewidth(2.0 if col == fired else 0.6)
            s.set_color(C_FIRE if col == fired else '#B9BEC6')

    def tag(ax, txt, col, y):
        ax.text(.03, y, txt, transform=ax.transAxes, fontsize=5.5,
                va='top', color=col)

    # --- waveform: this unit, its reference channel, and its neighbours
    if data['wf'] is not None:
        t = np.arange(len(data['wf'])) / FS * 1000
        pk = np.max(np.abs(data['wf'])) + 1e-12
        for o in others:
            if o.get('wf') is not None and len(o['wf']) == len(t):
                a_wf.plot(t, o['wf'] / pk, color=ocol[o['clu_id']], lw=1.0,
                          alpha=.85)
        if data['noise_wf'] is not None:
            a_wf.plot(t, data['noise_wf'] / pk, color=C_REF, lw=1.0, ls='--')
        a_wf.plot(t, data['wf'] / pk, color=C_UNIT, lw=1.6)
        a_wf.set_ylim(-1.45, 1.15)
        y = .97
        tag(a_wf, f"clu {data['clu_id']}", C_UNIT, y)
        for o in others:
            if o.get('wf') is not None:
                y -= .13
                tag(a_wf, f"clu {o['clu_id']}", ocol[o['clu_id']], y)
        if data['noise_wf'] is not None:
            tag(a_wf, 'reference ch', C_REF, y - .13)
    frame(a_wf, 'waveform')

    # --- ISI: within-cluster, so no other cluster belongs here
    st = data['times']
    if st.size > 2:
        isi = np.diff(np.sort(st)) * 1000
        isi = isi[isi > 0]
        if isi.size:
            bins = np.logspace(-1, np.log10(max(isi.max(), 10)), 40)
            a_isi.hist(isi, bins=bins, color=C_HIST, lw=0)
            a_isi.set_xscale('log')
            a_isi.axvspan(0.1, 1.0, color=C_FIRE, alpha=.13, lw=0)
    frame(a_isi, 'isi')

    # --- KS4 feature space: the unit (split by imposed group when the dip is
    # real), the neighbours, and the reference channel. Both scopes live here
    # because both rules that read feature space -- 4a against the reference
    # channel, 5a against a neighbour -- need their evidence visible.
    dp = rec_row.get('_dip_pval')
    split_is_real = _isfinite(dp) and dp < DIP_PVAL_THRESH
    res = None
    if data['feats'] is not None and len(data['feats']) >= 240:
        try:
            res = overmerge_verdict(data['feats'])
        except Exception:
            res = None
    # NO REFERENCE CHANNEL HERE. The screen's unit/noise clouds live in the
    # screen's own 3-D space, not KS4's flattened multichannel PCs -- drawing
    # them on these axes put two different feature spaces on one pair of axes,
    # and the reference cloud's scale squashed the unit to a dot. The
    # reference channel is a CHANNEL, not a cluster: it has no KS4 cluster
    # features, and it appears in this figure as a waveform.
    #
    # ALIGNED TO SHARED CHANNELS. KS4 stores every cluster's features on its
    # OWN local channel set, so column 0 is a different channel for different
    # clusters; plotting raw columns together compares two channels and calls
    # the result separation. aligned_features() restricts both to the channels
    # they share, which is the only way the two clouds are comparable. The
    # unit is drawn in the primary neighbour's alignment so one pair of axes
    # means one thing.
    fs_pair = data.get('fs_aligned')
    if fs_pair is not None:
        fa, fb, oid, n_shared = fs_pair
        a_fs.scatter(fb[:, 0], fb[:, 1], s=.6, color=ocol.get(oid, C_OTHER[0]),
                     alpha=.35, linewidths=0)
        lab = res.get('labels') if (res and split_is_real) else None
        if lab is not None and len(lab) == len(fa):
            lab = np.asarray(lab)
            a_fs.scatter(fa[lab == 0, 0], fa[lab == 0, 1], s=.6, color=C_UNIT,
                         alpha=.55, linewidths=0)
            a_fs.scatter(fa[lab == 1, 0], fa[lab == 1, 1], s=.6, color=C_SPLIT,
                         alpha=.55, linewidths=0)
        else:
            a_fs.scatter(fa[:, 0], fa[:, 1], s=.6, color=C_UNIT, alpha=.5,
                         linewidths=0)
        role = next((o['role'] for o in others if o['clu_id'] == oid), '')
        tag(a_fs, f'clu {oid} ({role})', ocol.get(oid, C_OTHER[0]), .97)
        tag(a_fs, f'{n_shared} shared channels', C_QUIET, .84)
    elif data['feats'] is not None and len(data['feats']):
        f = data['feats']
        a_fs.scatter(f[:, 0], f[:, 1], s=.6, color=C_UNIT, alpha=.5,
                     linewidths=0)
        tag(a_fs, 'no shared-channel neighbour', C_QUIET, .97)
    frame(a_fs, 'feature_space')

    # --- 1-D projection: purely within-cluster
    if res is not None and res.get('proj') is not None:
        proj = np.asarray(res['proj'], float)
        # THE TWO-GROUP COLOURING IS ALWAYS AVAILABLE AND USUALLY WRONG. The
        # split is imposed on every cluster whether or not it has two modes,
        # so colouring a unimodal one shows two populations the test just
        # ruled out. Only a significant dip earns the second colour.
        lab = res.get('labels') if split_is_real else None
        if lab is not None and len(lab) == len(proj) and len(set(lab)) > 1:
            lab = np.asarray(lab)
            bins = np.linspace(proj.min(), proj.max(), 34)
            a_prj.hist(proj[lab == 0], bins=bins, color=C_UNIT, lw=0)
            a_prj.hist(proj[lab == 1], bins=bins, color=C_SPLIT, lw=0)
        else:
            a_prj.hist(proj, bins=30, color=C_UNIT, lw=0)
        if _isfinite(dp):
            a_prj.text(.03, .97, f'dip p = {dp:.3f}', transform=a_prj.transAxes,
                       fontsize=5.5, va='top',
                       color=C_FIRE if split_is_real else C_QUIET)
        if rec_row.get('dip_basis'):
            a_prj.text(.97, .97, rec_row['dip_basis'], transform=a_prj.transAxes,
                       fontsize=4.5, color=C_QUIET, ha='right', va='top')
    else:
        a_prj.text(.5, .5, 'dip not evaluated', ha='center', va='center',
                   transform=a_prj.transAxes, fontsize=6, color=C_QUIET)
    frame(a_prj, 'projection')

    # --- amplitude over time: this unit against the same neighbours
    for o in others:
        if o.get('times') is not None and o['times'].size:
            a_amp.scatter(o['times'] / 60, o['amps'], s=.6,
                          color=ocol[o['clu_id']], alpha=.30, linewidths=0)
    if st.size:
        a_amp.scatter(st / 60, data['amps'], s=.6, color=C_UNIT, alpha=.5,
                      linewidths=0)
    af = rec_row.get('absent_frame_frac')
    if _isfinite(af) and af > 0:
        a_amp.text(.03, .97, f'absent {af:.0%}', transform=a_amp.transAxes,
                   fontsize=5.5, va='top',
                   color=C_FIRE if af > ABSENT_FRAME_FRAC_THRESH else C_QUIET)
    frame(a_amp, 'amplitude')

    # --- CCG against the partner the rule used, else the first neighbour
    prt = next((o for o in others if o.get('is_partner')),
               others[0] if others else None)
    if prt is not None and prt.get('times') is not None and st.size and prt['times'].size:
        # compute_ccg takes RAW SAMPLES and divides by fs itself; handing it
        # seconds compressed the recording into 20 ms and made every CCG a
        # triangle. Same window and bin as the rule, so the panel shows what
        # the gate saw.
        centers, counts = compute_ccg(st * FS, prt['times'] * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                  color=C_HIST, lw=0)
        a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.13, lw=0)
        tag(a_ccg, f"vs clu {prt['clu_id']} ({prt['role']})",
            ocol.get(prt['clu_id'], C_QUIET), .97)
    elif st.size > 2:
        centers, counts = compute_ccg(st * FS, st * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                  color=C_HIST, lw=0)
        a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.13, lw=0)
        tag(a_ccg, 'autocorrelogram (no neighbour)', C_QUIET, .97)
    frame(a_ccg, 'ccg')

    for ax in axes:
        ax.tick_params(labelsize=5, length=2, pad=1)
        if not show_xlabels:
            ax.set_xticklabels([])
        ax.set_yticks([])
