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

Columns, left to right -- the within-cluster block, then the between-cluster
block, so the colour grammar reads in blocks:

  waveform | ISI | within-cluster PC | 1-D projection | between-cluster PC | amplitude | CCG
    all      r2      r3a (imposed)        r3a              r4a / 5a            r3b     r1/5b/5c

TWO FEATURE SPACES, because they answer different questions with the same
numbers. The within-cluster panel colours ONE cluster's spikes by the split
the dip test imposes on it; the between-cluster panel puts this cluster
against a neighbour on the channels they share. Collapsing them into one
panel is what made rule 4a's evidence disappear earlier.

THE WAVEFORM PANEL HOLDS THREE KINDS OF TRACE, all scaled by THIS unit's
peak -- which is the point, since that shared scale is what the printed
peak:reference ratio measures:

    blue solid    this cluster on its peak channel
    grey dashed   this cluster on its REFERENCE channel (a channel, not a
                  cluster -- the one layer 1 measured isolation against)
    red dotted    a neighbouring CLUSTER on the same peak channel

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
                    # other_clu holds 1-INDEXED ms_ids, not cluster ids --
                    # compute_isolation compares them against `good_clus_0 + 1`
                    # and the SortData files are named the same way. Taking
                    # them as cluster ids drew the wrong neighbour in every
                    # between-cluster panel, and for clu 46 drew the cluster
                    # against ITSELF.
                    ent = wf_by_clu.setdefault(int(cid) - 1, [])
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
    neigh = [(c, w) for c, w in neigh if c != clu_id and c >= 0]
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
        # the key is 'other', not 'clu' -- ccg_accusers uses 'clu' and the two
        # schemas differ, which only stayed hidden because a duplicate sets
        # redundant_of first and never reaches this branch
        p = rec_row['overlap_pairs'][0]
        pid = int(p.get('other', p.get('clu')) if isinstance(p, dict) else p)
        why = 'overlapping'
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


def _running(t, y, n_bins=26):
    """Median of y in equal time bins -- the trace laid over the dots."""
    if t.size < 8:
        return None, None
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    mids, med = [], []
    for i in range(n_bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() >= 3:
            mids.append((edges[i] + edges[i + 1]) / 2)
            med.append(np.median(y[m]))
    return (np.asarray(mids), np.asarray(med)) if mids else (None, None)


def draw_row(axes, rec_row, data, show_xlabels=False):
    """Draw one cluster across seven axes:
    waveform, ISI, within-cluster PC, projection, between-cluster PC,
    amplitude, CCG."""
    a_wf, a_isi, a_win, a_prj, a_btw, a_amp, a_ccg = axes
    fired = rec_row.get('_fires')
    others = data.get('others') or []
    # ONE comparison cluster for the whole row. The rule's partner and layer
    # 1's same-channel neighbour are not always the same cluster -- for one of
    # the over-merge picks the accuser is 152 while the screen's neighbour is
    # 153 -- so letting each panel pick its own would silently compare
    # different pairs across the row.
    primary = next((o for o in others if o.get('is_partner')),
                   others[0] if others else None)
    # COLOUR BY ROLE, NOT BY POSITION IN THE LIST. Keying off enumerate()
    # meant the comparison cluster came out red in one row and pink in the
    # next purely because it happened to be second in the neighbour list --
    # the reader has no way to know that, so the colour stopped meaning
    # anything. The cluster the row is ABOUT is always C_OTHER[0].
    ocol = {}
    if primary is not None:
        ocol[primary['clu_id']] = C_OTHER[0]
    for o in others:
        ocol.setdefault(o['clu_id'], C_OTHER[1])
    pcol = ocol.get(primary['clu_id'], C_OTHER[0]) if primary else C_OTHER[0]

    def frame(ax, col):
        for sp_ in ax.spines.values():
            sp_.set_linewidth(2.0 if col == fired else 0.6)
            sp_.set_color(C_FIRE if col == fired else '#B9BEC6')

    def tag(ax, txt, col, y, size=5.5):
        ax.text(.03, y, txt, transform=ax.transAxes, fontsize=size,
                va='top', color=col)

    # --- waveform. ALL traces share this unit's peak as the scale: that is
    # what makes the peak:reference ratio readable off the panel, and it is
    # how the ratio was defined in the first place.
    if data['wf'] is not None:
        t = np.arange(len(data['wf'])) / FS * 1000
        peak = np.max(np.abs(data['wf'])) + 1e-12
        if primary is not None and primary.get('wf') is not None \
                and len(primary['wf']) == len(t):
            a_wf.plot(t, primary['wf'] / peak, color=pcol, lw=1.1, ls=':')
        if data['noise_wf'] is not None:
            a_wf.plot(t, data['noise_wf'] / peak, color=C_REF, lw=1.1, ls='--')
        a_wf.plot(t, data['wf'] / peak, color=C_UNIT, lw=1.8)
        # Every trace is scaled by THIS unit's peak -- that shared scale is
        # what the printed ratio means, so it stays. But a neighbour can be
        # several times larger, and a fixed y-limit then cropped it into a
        # line that ran off the panel. Fit the limits to what is drawn.
        shown = [data['wf'] / peak]
        if data['noise_wf'] is not None:
            shown.append(np.asarray(data['noise_wf']) / peak)
        if primary is not None and primary.get('wf') is not None \
                and len(primary['wf']) == len(t):
            shown.append(np.asarray(primary['wf']) / peak)
        allv = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in shown])
        lo, hi = float(np.min(allv)), float(np.max(allv))
        pad = 0.10 * (hi - lo + 1e-9)
        # headroom at the top for the three legend lines
        a_wf.set_ylim(lo - pad, hi + pad + 0.55 * (hi - lo + 1e-9))
        y = .97
        tag(a_wf, f"clu {data['clu_id']} peak ch", C_UNIT, y); y -= .125
        if data['noise_wf'] is not None:
            nz = np.max(np.abs(data['noise_wf']))
            if np.isfinite(nz) and nz > 0:
                tag(a_wf, f'reference ch  ({peak / nz:.1f}x)', C_REF, y)
            else:
                tag(a_wf, 'reference ch', C_REF, y)
            y -= .125
        if primary is not None and primary.get('wf') is not None:
            tag(a_wf, f"clu {primary['clu_id']} ({primary['role']})", pcol, y)
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
            a_isi.axvspan(0.1, 1.0, color=C_FIRE, alpha=.16, lw=0)
    frame(a_isi, 'isi')

    dp = rec_row.get('_dip_pval')
    split_is_real = _isfinite(dp) and dp < DIP_PVAL_THRESH
    res = None
    if data['feats'] is not None and len(data['feats']) >= 240:
        try:
            res = overmerge_verdict(data['feats'])
        except Exception:
            res = None
    lab = np.asarray(res['labels']) if (res and split_is_real
                                        and res.get('labels') is not None) else None

    # --- within-cluster PC: ONE cluster, coloured by the split the dip
    # imposes on it. No alignment question arises -- these are all the same
    # cluster's spikes on its own channels.
    if data['feats'] is not None and len(data['feats']):
        fw = data['feats']
        if lab is not None and len(lab) == len(fw):
            a_win.scatter(fw[lab == 0, 0], fw[lab == 0, 1], s=2.6,
                          color=C_UNIT, alpha=.85, linewidths=0)
            a_win.scatter(fw[lab == 1, 0], fw[lab == 1, 1], s=2.6,
                          color=C_SPLIT, alpha=.85, linewidths=0)
            tag(a_win, 'imposed split', C_SPLIT, .97)
        else:
            a_win.scatter(fw[:, 0], fw[:, 1], s=2.6, color=C_UNIT,
                          alpha=.85, linewidths=0)
            tag(a_win, 'one mode', C_QUIET, .97)
    frame(a_win, 'within_pc')

    # --- 1-D projection: the axis the dip actually ran on
    if res is not None and res.get('proj') is not None:
        proj = np.asarray(res['proj'], float)
        # THE TWO-GROUP COLOURING IS ALWAYS AVAILABLE AND USUALLY WRONG. The
        # split is imposed on every cluster whether or not it has two modes,
        # so colouring a unimodal one shows two populations the test just
        # ruled out. Only a significant dip earns the second colour.
        if lab is not None and len(lab) == len(proj) and len(set(lab)) > 1:
            bins = np.linspace(proj.min(), proj.max(), 34)
            a_prj.hist(proj[lab == 0], bins=bins, color=C_UNIT, lw=0)
            a_prj.hist(proj[lab == 1], bins=bins, color=C_SPLIT, lw=0)
        else:
            a_prj.hist(proj, bins=30, color=C_UNIT, lw=0)
        if _isfinite(dp):
            tag(a_prj, f'dip p = {dp:.3f}',
                C_FIRE if split_is_real else C_QUIET, .97)
        if rec_row.get('dip_basis'):
            a_prj.text(.97, .97, rec_row['dip_basis'], transform=a_prj.transAxes,
                       fontsize=4.5, color=C_QUIET, ha='right', va='top')
    else:
        a_prj.text(.5, .5, 'dip not evaluated', ha='center', va='center',
                   transform=a_prj.transAxes, fontsize=6, color=C_QUIET)
    frame(a_prj, 'projection')

    # --- between-cluster PC, ALIGNED TO SHARED CHANNELS. KS4 stores every
    # cluster's features on its OWN local channel set, so column 0 is a
    # different channel for different clusters; plotting raw columns together
    # compares two channels and calls the result separation.
    fs_pair = data.get('fs_aligned')
    if fs_pair is not None:
        fa, fb, oid, n_shared = fs_pair
        a_btw.scatter(fb[:, 0], fb[:, 1], s=2.6, color=ocol.get(oid, C_OTHER[0]),
                      alpha=.80, linewidths=0)
        a_btw.scatter(fa[:, 0], fa[:, 1], s=2.6, color=C_UNIT, alpha=.80,
                      linewidths=0)
        role = next((o['role'] for o in others if o['clu_id'] == oid), '')
        tag(a_btw, f'vs clu {oid} ({role})', ocol.get(oid, C_OTHER[0]), .97)
        tag(a_btw, f'{n_shared} shared channels', C_QUIET, .845, size=5)
        # Whether the partner sits on the unit's OWN peak channel decides
        # whether layer 1 could see this comparison at all -- it only ever
        # references same-channel clusters. For duplicates it usually cannot:
        # 39.6% of their partners are on another channel, because re-detection
        # is spatial. Saying so turns a caveat into the row's lesson.
        sc = rec_row.get('_partner_same_channel')
        if sc is False:
            tag(a_btw, 'different peak channel', C_FIRE, .72, size=5)
            tag(a_btw, 'layer 1 never compares these', C_QUIET, .60, size=4.5)
    else:
        a_btw.text(.5, .5, 'no shared-channel neighbour', ha='center',
                   va='center', transform=a_btw.transAxes, fontsize=6,
                   color=C_QUIET)
    frame(a_btw, 'between_pc')

    # --- amplitude over time, with a median trace per cluster so a drift or
    # a gap reads without squinting at the cloud
    if primary is not None and primary.get('times') is not None \
            and primary['times'].size:
        a_amp.scatter(primary['times'] / 60, primary['amps'], s=2.2,
                      color=pcol, alpha=.55, linewidths=0)
        mx, my = _running(primary['times'] / 60, primary['amps'])
        if mx is not None:
            a_amp.plot(mx, my, color=pcol, lw=1.3)
    if st.size:
        a_amp.scatter(st / 60, data['amps'], s=2.2, color=C_UNIT, alpha=.65,
                      linewidths=0)
        mx, my = _running(st / 60, data['amps'])
        if mx is not None:
            a_amp.plot(mx, my, color=C_UNIT, lw=1.6)
    af = rec_row.get('absent_frame_frac')
    if _isfinite(af) and af > 0:
        tag(a_amp, f'absent {af:.0%}',
            C_FIRE if af > ABSENT_FRAME_FRAC_THRESH else C_QUIET, .97)
    frame(a_amp, 'amplitude')

    # --- CCG against that SAME comparison cluster
    if primary is not None and primary.get('times') is not None \
            and st.size and primary['times'].size:
        # compute_ccg takes RAW SAMPLES and divides by fs itself; handing it
        # seconds compressed the recording into 20 ms and made every CCG a
        # triangle. Same window and bin as the rule, so the panel shows what
        # the gate saw.
        centers, counts = compute_ccg(st * FS, primary['times'] * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                  color=C_HIST, lw=0)
        a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.16, lw=0)
        # THE LAG IS THE TEST, NOT THE EXCESS. Re-detections peak OFF zero --
        # the duplicate exemplars sit at +/-0.25 to 1.35 ms -- which is why a
        # zero-centred coincidence test (SpikeInterface's delta_time=0.4)
        # catches none of them. Over the +/-25 ms the gate uses, a 1.25 ms
        # offset is indistinguishable from zero by eye, so the panel has to
        # say where the peak is or it shows a duplicate and an over-split as
        # the same picture. Refractory dip vs coincident excess is then
        # readable too.
        if counts.size and counts.max() > 0:
            ml = float(centers[int(np.argmax(counts))])
            a_ccg.axvline(ml, color=C_FIRE, lw=0.8, ls=':')
            flank = counts[(np.abs(centers) > 5) & (np.abs(centers) <= 20)]
            ctr = counts[np.abs(centers) <= 1.0]
            ratio = (ctr.mean() / max(flank.mean(), 1)) if flank.size and ctr.size else float('nan')
            word = ('refractory dip' if ratio < 0.3
                    else 'coincident excess' if ratio > 1.5 else 'flat')
            a_ccg.text(.97, .97, f'peak {ml:+.2f} ms\n{word}',
                       transform=a_ccg.transAxes, fontsize=5, ha='right',
                       va='top', color=C_QUIET)
        tag(a_ccg, f"vs clu {primary['clu_id']} ({primary['role']})", pcol, .97)
    elif st.size > 2:
        centers, counts = compute_ccg(st * FS, st * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                  color=C_HIST, lw=0)
        a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.16, lw=0)
        tag(a_ccg, 'autocorrelogram (no neighbour)', C_QUIET, .97)
    frame(a_ccg, 'ccg')

    for ax in axes:
        ax.tick_params(labelsize=5, length=2, pad=1)
        if not show_xlabels:
            ax.set_xticklabels([])
        ax.set_yticks([])
