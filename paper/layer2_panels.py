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

from matplotlib.ticker import MaxNLocator

from unitassay.curate.bimodality import overmerge_verdict
from unitassay.curate.ccg import CCG_BIN_MS, CCG_WINDOW_MS, compute_ccg
from unitassay.curate.constants import ABSENT_FRAME_FRAC_THRESH, DIP_PVAL_THRESH
from unitassay.curate.sortdata import aligned_features
from unitassay.screen.io.ks_loader import load_ks_dir

N_PC = 3
FS = 30000.0

# Red for a cluster INSIDE layer 1's pool -- same peak channel, one of the
# clusters its L-ratio was measured against. Pink for one OUTSIDE it, which
# only layer 2 ever reaches. Agrita: a duplicate's partner is on a different
# channel by definition, so it should not be drawn in the colour that means
# "layer 1 compared these". Both hues validated all-pairs for CVD against the
# unit blue and the split orange (worst pair dE 15.6 deutan).
# PASTEL FILLS, FULL-STRENGTH LINES. Agrita: the saturated hues are harsh,
# and Figure 3 is easy on the eyes. Sampling Figure 3's own panels shows why
# -- its dot cores ARE saturated (#30489C, #B41824), but the dots are small
# over white, so the FIELD reads light while every stroke stays crisp. These
# panels have far denser clouds, so the same trick does not carry: the fills
# are tints (20% white) and the lines, waveforms and labels keep the full
# hue, which is what has to hold at 5 pt.
#
# The four tints pass every check all-pairs -- lightness band, chroma floor,
# CVD separation (worst 9.5 dE deutan) and the normal-vision floor.
C_UNIT = '#6175B6'          # this cluster -- tint of the manuscript blue
C_SPLIT = '#3BBB70'         # the minority group the imposed split peels off
C_SAME_POOL = '#CE4679'     # another cluster INSIDE layer 1's pool
C_OFF_POOL = '#E692BE'      # one outside it -- lighter, same family
C_OTHER = (C_SAME_POOL, C_OFF_POOL)

L_UNIT = '#3953A4'          # full strength, for strokes and text
L_SPLIT = '#0AAA4B'
L_POOL = {C_SAME_POOL: '#C2185B', C_OFF_POOL: '#D4569B'}

C_REF = '#9AA3AE'           # the reference channel: a channel, not a cluster
C_FRAME = '#14213D'         # the panel whose evidence the rule read
C_WASH = '#EFF2F7'          # its background, instead of a coloured border
C_QUIET = '#5B6470'
C_HIST = '#9098A3'
C_REFRAC = '#BF1E2D'        # the refractory window, shaded red


def _isfinite(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x))


def _flat_feats(sp, clu_id, with_mask=False):
    idx = np.where(sp.clu == clu_id)[0]
    if sp.pc_feat is None or not idx.size:
        return (None, None) if with_mask else None
    npc = min(N_PC, sp.pc_feat.shape[1])
    flat = sp.pc_feat[idx, :npc, :].reshape(idx.size, -1)
    # KEEP THE MASK. The imposed-split labels are computed on the rows that
    # survive here, so the spike times and amplitudes cannot be coloured by
    # those labels unless we know which spikes they correspond to.
    keep = np.all(np.isfinite(flat), axis=1)
    out = flat[keep]
    out = out if out.size else None
    return (out, keep) if with_mask else out


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
           'feats': None, 'wf': None, 'noise_wf': None,
           'screen_unit': None, 'screen_noise': None, 'others': []}

    out['feats'], out['split_mask'] = _flat_feats(sp, clu_id, with_mask=True)

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
        # THE SAME SPIKES, IN THE SAME ORDER, so the imposed-split labels
        # computed on the full feature matrix can colour the aligned points.
        # aligned_features() drops non-finite rows on the aligned columns and
        # load_row() drops them on the full ones; the two masks are not the
        # same, so the labels cannot simply be indexed onto fa. Recomputing
        # the split on the aligned subset would be a DIFFERENT test from the
        # one the record reports, so instead the unit's aligned points are
        # rebuilt from the rows that survived the FULL mask.
        ind_a, ind_b = sp.pc_feat_ind[data['clu_id']], sp.pc_feat_ind[prt['clu_id']]
        shared = np.intersect1d(ind_a, ind_b)
        cols_a = np.array([np.where(ind_a == ch)[0][0] for ch in shared])
        idx_a = np.where(sp.clu == data['clu_id'])[0]
        npc = min(N_PC, sp.pc_feat.shape[1])
        full = sp.pc_feat[idx_a][:, :npc, :].reshape(len(idx_a), -1)
        keep = np.all(np.isfinite(full), axis=1)
        al = sp.pc_feat[idx_a][keep][:, :npc, :][:, :, cols_a]
        data['fs_unit_registered'] = al.reshape(int(keep.sum()), -1)
    return data


def add_rule_partner(data, rec_row, sp_cache_entry, ks_dir=None,
                     unit_peak_ch=None):
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
            'clu_id': pid, 'wf': partner_template_on(ks_dir, pid, unit_peak_ch),
            'role': why, 'is_partner': True,
            'times': sp.st[oidx], 'amps': sp.temp_scaling_amps[oidx],
            'feats': _flat_feats(sp, pid)})
    return data


_TEMPL = {}


def unit_peak_channel(ks_dir, clu_id):
    """The channel the screen calls this cluster's peak -- max template energy."""
    t = _templates(ks_dir)
    if t is None or clu_id >= len(t):
        return None
    return int(np.argmax(np.ptp(np.asarray(t[clu_id]), axis=0)))


def _templates(ks_dir):
    if ks_dir not in _TEMPL:
        try:
            _TEMPL[ks_dir] = np.load(Path(ks_dir) / 'templates.npy', mmap_mode='r')
        except Exception:
            _TEMPL[ks_dir] = None
    return _TEMPL[ks_dir]


def partner_template_on(ks_dir, clu_id, channel):
    """The partner's template ON THE UNIT'S PEAK CHANNEL.

    A duplicate's partner usually sits on a DIFFERENT peak channel -- 39.6% of
    them do, because re-detection is spatial -- so it is not in the screen's
    same-channel neighbour list and has no stored waveform. Reading it off the
    templates keeps the panel populated, and putting it on the UNIT's channel
    is what makes the comparison mean anything: the claim is that the same
    event was detected twice, which is a claim about one channel.
    """
    t = _templates(ks_dir)
    if t is None or channel is None or clu_id >= len(t):
        return None
    try:
        return np.asarray(t[clu_id])[:, channel]
    except Exception:
        return None


def _rebin(centers, counts, k):
    """Pair up k adjacent bins for display."""
    n = (len(counts) // k) * k
    c = np.asarray(counts[:n], float).reshape(-1, k).sum(1)
    x = np.asarray(centers[:n], float).reshape(-1, k).mean(1)
    return x, c


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
    # Colour says WHICH POOL the comparison cluster came from, not where it
    # sat in a list: red inside layer 1's same-channel pool, pink outside it.
    off_pool = rec_row.get('_partner_same_channel') is False
    ocol = {}
    if primary is not None:
        ocol[primary['clu_id']] = C_OFF_POOL if off_pool else C_SAME_POOL
    for o in others:
        ocol.setdefault(o['clu_id'], C_SAME_POOL)
    pcol = ocol.get(primary['clu_id'], C_SAME_POOL) if primary else C_SAME_POOL

    def frame(ax, col):
        # The evidence the rule read gets a pale wash and a darker edge. A
        # thick coloured border competed with the data for attention and put a
        # fifth hue on the page.
        hot = (col == fired)
        if hot:
            ax.set_facecolor(C_WASH)
        for side, sp_ in ax.spines.items():
            sp_.set_visible(side in ('left', 'bottom') or hot)
            sp_.set_linewidth(1.1 if hot else 0.6)
            sp_.set_color(C_FRAME if hot else '#C7CCD4')

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
            a_wf.plot(t, primary['wf'] / peak, color=L_POOL.get(pcol, pcol), lw=1.2, ls=':')
        if data['noise_wf'] is not None:
            a_wf.plot(t, data['noise_wf'] / peak, color=C_REF, lw=1.1, ls='--')
        a_wf.plot(t, data['wf'] / peak, color=L_UNIT, lw=1.8)
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
        tag(a_wf, f"clu {data['clu_id']} peak ch", L_UNIT, y); y -= .125
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
            a_isi.axvspan(0.1, 1.0, color=C_REFRAC, alpha=.15, lw=0)
    frame(a_isi, 'isi')

    dp = rec_row.get('_dip_pval')
    # A SIGNIFICANT DIP IS NOT AN ADOPTED VERDICT. 1,254 clusters in the corpus
    # carry dip p < 0.05 that curate did NOT call over-merged -- 1,007 it left
    # `ambiguous` and 247 it judged `not_overmerged`, usually because the split
    # peeled off a negligible satellite. Colouring on the raw p would have
    # shown the reader two populations the tool explicitly declined to claim,
    # which is the same mislabel as reading an absent dominant_source as a
    # clean result. The verdict decides; the p-value is only reported.
    verdict = rec_row.get('ldl_verdict')
    dip_sig = _isfinite(dp) and dp < DIP_PVAL_THRESH
    split_is_real = (verdict == 'overmerged')
    # THE SPLIT IS DRAWN ON EVERY ROW, and that is the point (Agrita). The
    # two-group split is imposed on every cluster whether or not it has two
    # modes; showing it only where it was adopted hides that, and leaves the
    # reader to assume the colours appeared because the structure did. Drawn
    # everywhere, the non-over-merge rows show the two groups sitting on top
    # of each other -- the split exists, it just does not separate -- and the
    # over-merge row is the one where it comes apart.
    #
    # What still depends on the VERDICT is the claim, never the colour: the
    # frame highlight, and the label beside the p-value. 1,254 clusters carry
    # dip p < 0.05 that curate did not adopt, so the panel says which.
    res = None
    if data['feats'] is not None and len(data['feats']) >= 240:
        try:
            res = overmerge_verdict(data['feats'])
        except Exception:
            res = None
    lab = (np.asarray(res['labels'])
           if (res and res.get('labels') is not None) else None)
    # WHICH GROUP IS 0 IS ARBITRARY. The splitter labels its two groups in
    # whatever order the fit produced, so the same colour meant "most of the
    # cluster" on one row and "the peeled-off part" on the next -- on the
    # excess-at-zero row orange was the majority, which reads as the cluster
    # being mostly the minority population. Blue is the majority everywhere
    # now, orange the smaller group.
    if lab is not None and lab.size and (lab == 1).sum() > (lab == 0).sum():
        lab = 1 - lab
    # WHERE ORANGE IS LICENSED (Agrita). The within-cluster PC and the 1-D
    # projection are ABOUT the imposed split and say so, so they show it on
    # every row -- that is how a reader sees it fail to separate. The
    # between-cluster panel and the amplitude trace are about this cluster in
    # relation to others, and coluring an unadopted split there would assert a
    # second population as an entity. Only the adopted verdict licenses that,
    # so elsewhere the unit is drawn as one cluster.
    lab_claim = lab if split_is_real else None

    # --- within-cluster PC: ONE cluster, coloured by the split the dip
    # imposes on it. No alignment question arises -- these are all the same
    # cluster's spikes on its own channels.
    if data['feats'] is not None and len(data['feats']):
        fw = data['feats']
        if lab is not None and len(lab) == len(fw):
            a_win.scatter(fw[lab == 0, 0], fw[lab == 0, 1], s=2.6,
                          color=C_UNIT, alpha=.92, linewidths=0)
            a_win.scatter(fw[lab == 1, 0], fw[lab == 1, 1], s=2.6,
                          color=C_SPLIT, alpha=.92, linewidths=0)
            tag(a_win, 'imposed split', L_SPLIT if split_is_real else C_QUIET,
                .97)
        else:
            a_win.scatter(fw[:, 0], fw[:, 1], s=3.0, color=C_UNIT,
                          alpha=.92, linewidths=0)
            tag(a_win, 'split not computed', C_QUIET, .97)
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
            note = f'dip p = {dp:.3f}'
            if dip_sig and not split_is_real:
                note += f'  ({verdict})'
            tag(a_prj, note, L_SPLIT if split_is_real else C_QUIET, .97)
        if rec_row.get('dip_basis'):
            # bottom right: the basis string runs long and collided with the
            # p-value once the verdict was appended to it
            # first token only: the full string ('negligible_satellite+
            # large_tested') runs the width of the panel and lands on the
            # tick labels
            a_prj.text(.97, .88, str(rec_row['dip_basis']).split('+')[0],
                       transform=a_prj.transAxes, fontsize=4.5,
                       color=C_QUIET, ha='right', va='top')
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
        a_btw.scatter(fb[:, 0], fb[:, 1], s=3.0, color=ocol.get(oid, C_OTHER[0]),
                      alpha=.92, linewidths=0)
        reg = data.get('fs_unit_registered')
        if lab_claim is not None and reg is not None and len(reg) == len(lab_claim):
            # An over-merge is two populations in ONE cluster, so the question
            # this panel has to answer is WHICH of them sits on the neighbour.
            # Drawn in one colour it cannot answer it.
            a_btw.scatter(reg[lab_claim == 0, 0], reg[lab_claim == 0, 1],
                          s=3.0, color=C_UNIT, alpha=.92, linewidths=0)
            a_btw.scatter(reg[lab_claim == 1, 0], reg[lab_claim == 1, 1],
                          s=3.0, color=C_SPLIT, alpha=.92, linewidths=0)
        else:
            a_btw.scatter(fa[:, 0], fa[:, 1], s=3.0, color=C_UNIT, alpha=.80,
                          linewidths=0)
        role = next((o['role'] for o in others if o['clu_id'] == oid), '')
        tag(a_btw, f'vs clu {oid} ({role})', ocol.get(oid, C_OTHER[0]), .97)
        tag(a_btw, f'{n_shared} shared channels', C_QUIET, .845, size=5)
        # Whether the partner sits on the unit's OWN peak channel decides
        # whether layer 1 could see this comparison at all -- it only ever
        # references same-channel clusters. For duplicates it usually cannot:
        # 39.6% of their partners are on another channel, because re-detection
        # is spatial. Saying so turns a caveat into the row's lesson.
        if rec_row.get('_partner_same_channel') is False:
            tag(a_btw, 'different peak channel', C_OFF_POOL, .72, size=5)
            tag(a_btw, "outside layer 1's pool", C_QUIET, .60, size=4.5)
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
                      color=pcol, alpha=.85, linewidths=0)
        mx, my = _running(primary['times'] / 60, primary['amps'])
        if mx is not None:
            a_amp.plot(mx, my, color=L_POOL.get(pcol, pcol), lw=1.8)
    msk = data.get('split_mask')
    if (st.size and lab_claim is not None and msk is not None
            and int(np.sum(msk)) == len(lab_claim)):
        # The two halves of an over-merge are two populations in time as well
        # as in feature space, and whether they differ in amplitude is part of
        # what the row is for. Drawn in one colour the panel cannot say.
        ts, as_ = st[msk] / 60, np.asarray(data['amps'])[msk]
        for g, col in ((0, C_UNIT), (1, C_SPLIT)):
            m = (lab_claim == g)
            if m.sum():
                a_amp.scatter(ts[m], as_[m], s=2.6, color=col, alpha=.65,
                              linewidths=0)
                gx, gy = _running(ts[m], as_[m])
                if gx is not None:
                    # a bare line disappears into its own point cloud; the
                    # white halo is what makes each trace readable
                    a_amp.plot(gx, gy, color=(L_UNIT if col == C_UNIT else L_SPLIT), lw=1.8)
    elif st.size:
        a_amp.scatter(st / 60, data['amps'], s=2.6, color=C_UNIT, alpha=.85,
                      linewidths=0)
        mx, my = _running(st / 60, data['amps'])
        if mx is not None:
            a_amp.plot(mx, my, color=L_UNIT, lw=1.8)
    af = rec_row.get('absent_frame_frac')
    if _isfinite(af) and af > 0:
        tag(a_amp, f'absent {af:.0%}',
            C_FRAME if af > ABSENT_FRAME_FRAC_THRESH else C_QUIET, .97)
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
        # WIDER BARS. The gate's 0.5 ms bins give 100 spikes of noise across
        # the window, in which an empty centre is invisible; paired into 1 ms
        # bins the silhouette reads, and a re-detection's lag is still
        # resolved. The gate's own resolution is unchanged -- the modal lag
        # printed below is measured on the 0.5 ms bins.
        centers, counts = _rebin(centers, counts, 4)      # 0.5 -> 2 ms bins
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]) * 0.92,
                  color=C_HIST, lw=0, alpha=.95)
        a_ccg.set_ylim(0, max(1.0, float(counts.max()) * 1.22))
        a_ccg.axvspan(-1, 1, color=C_REFRAC, alpha=.18, lw=0)
        # DRAW THE LEVEL THE CENTRE IS JUDGED AGAINST. The statistic compares
        # the +-1 ms window with the 5-20 ms flanks, and at three counts a bin
        # an empty centre is invisible against a noisy silhouette -- the dip
        # row read as noise while the duplicate's needle read instantly. With
        # the flank mean drawn, a hole below the line and a spike above it are
        # both obvious, and it is the comparison the test actually makes.
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
            flank = counts[(np.abs(centers) > 5) & (np.abs(centers) <= 20)]
            ctr = counts[np.abs(centers) <= 1.0]
            ratio = (ctr.mean() / max(flank.mean(), 1)) if flank.size and ctr.size else float('nan')
            is_dip = ratio < 0.3 and int(flank.sum()) >= 60
            if is_dip:
                # A DIPPING CORRELOGRAM HAS NO MODAL LAG WORTH QUOTING. Its
                # tallest bin is wherever noise happened to pile up -- for the
                # 5c exemplar that is -18.25 ms -- so printing it as "peak"
                # invites the reader to interpret a fluctuation. What the panel
                # is showing is the EMPTY centre, so report that instead.
                note = f'refractory dip\ndip ratio {ratio:.2f}'
            else:
                a_ccg.axvline(ml, color=C_FRAME, lw=0.8, ls=':')
                # FOR A DUPLICATE, QUOTE THE CRITERION THAT WAS APPLIED. The
                # dip ratio averages over +-1 ms, so a re-detection's needle --
                # one 0.5 ms bin -- is diluted to about 1.3 and comes out
                # labelled 'flat' on a panel where the spike is the most
                # obvious thing in the row. The rule never used that statistic
                # here: it used overlap and modal lag, so those are what the
                # panel reports.
                ov = None
                for q in (rec_row.get('overlap_pairs') or []):
                    if isinstance(q, dict) and int(q.get('other', -1)) == primary['clu_id']:
                        ov = q
                        break
                if ov is not None:
                    note = (f"overlap {float(ov.get('overlap', float('nan'))):.0%}\n"
                            f"lag {float(ov.get('lag_ms', float('nan'))):+.2f} ms")
                else:
                    word = 'coincident excess' if ratio > 1.5 else 'flat'
                    note = f'peak {ml:+.2f} ms\n{word}'
            a_ccg.text(.97, .97, note, transform=a_ccg.transAxes, fontsize=5,
                       ha='right', va='top', color=C_QUIET)
        tag(a_ccg, f"vs clu {primary['clu_id']} ({primary['role']})", pcol, .97)
    elif st.size > 2:
        centers, counts = compute_ccg(st * FS, st * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        # WIDER BARS. The gate's 0.5 ms bins give 100 spikes of noise across
        # the window, in which an empty centre is invisible; paired into 1 ms
        # bins the silhouette reads, and a re-detection's lag is still
        # resolved. The gate's own resolution is unchanged -- the modal lag
        # printed below is measured on the 0.5 ms bins.
        centers, counts = _rebin(centers, counts, 4)      # 0.5 -> 2 ms bins
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]) * 0.92,
                  color=C_HIST, lw=0, alpha=.95)
        a_ccg.set_ylim(0, max(1.0, float(counts.max()) * 1.22))
        a_ccg.axvspan(-1, 1, color=C_REFRAC, alpha=.18, lw=0)
        tag(a_ccg, 'autocorrelogram (no neighbour)', C_QUIET, .97)
    frame(a_ccg, 'ccg')

    for ax in axes:
        ax.tick_params(labelsize=5, length=2, pad=1)
        if not show_xlabels:
            ax.set_xticklabels([])
        ax.set_yticks([])
    AXLAB = ((a_wf, 'time (ms)', 'norm. amplitude'),
             (a_isi, 'ISI (ms)', 'count'),
             (a_win, 'PC 1', 'PC 2'),
             (a_prj, 'LDA projection', 'count'),
             (a_btw, 'PC 1 (shared ch)', 'PC 2'),
             (a_ccg, 'lag (ms)', 'coincidences'))
    for ax, xl, yl in AXLAB:
        ax.set_xlabel(xl, fontsize=5.5, labelpad=1)
        ax.set_ylabel(yl, fontsize=5.5, labelpad=1)
        ax.tick_params(labelsize=4.5, length=2, pad=1, labelleft=True,
                       labelbottom=True)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        if ax is not a_isi:                       # log axis picks its own
            ax.xaxis.set_major_locator(MaxNLocator(4))
        for t in ax.get_xticklabels():
            t.set_visible(True)

    # EVERY ROW GETS ITS OWN AMPLITUDE AXES. Recording length and amplitude
    # scale differ from row to row -- 6 to 37 minutes across these six -- so
    # tick labels only on the bottom row would be read as applying to all of
    # them. This panel carries its own, always.
    a_amp.tick_params(labelsize=5, length=2, pad=1, labelleft=True,
                      labelbottom=True)
    a_amp.yaxis.set_major_locator(MaxNLocator(3))
    a_amp.xaxis.set_major_locator(MaxNLocator(4))
    for lab_ in a_amp.get_xticklabels():
        lab_.set_visible(True)
    a_amp.set_xlabel('time (min)', fontsize=5.5, labelpad=1)
    a_amp.set_ylabel('amplitude (a.u.)', fontsize=5.5, labelpad=1)
