"""Figure 8 rows: one cluster drawn across the six columns of evidence.

THE RECORD DECIDES, THIS MODULE ONLY DRAWS. Every statistic shown beside a
panel is read from the stored curate() record. Recomputing a verdict here is
how the earlier rule figures ended up captioned with one test's answer and
drawn with another's, twice. Geometry (the LDA axis, the group labels) is
recomputed because it is not stored; verdicts never are.

Columns, in sieve order, so reading a row left to right is reading panel A
top to bottom:

    waveform | ISI | 1-D projection | amplitude | feature space | CCG
      ctx      r2        r3a            r3b          r4a/5a      r1/5b/5c
"""
import json
import math
from pathlib import Path

import numpy as np

from unitassay.curate.constants import (ABSENT_FRAME_FRAC_THRESH,
                                        DIP_PVAL_THRESH, ccg_z_threshold)
from unitassay.curate.ccg import CCG_BIN_MS, CCG_WINDOW_MS, compute_ccg
from unitassay.curate.bimodality import overmerge_verdict
from unitassay.curate.sortdata import sortdata_path
from unitassay.screen.io.ks_loader import load_ks_dir

N_PC = 3
FS = 30000.0
# One colour per role, held across every row so a colour means the same thing
# in all of them.
C_UNIT = '#1F4E9C'
C_PARTNER = '#C2185B'
C_REF = '#8A8F98'
C_FIRE = '#D55E00'
C_QUIET = '#5B6470'
C_G2 = '#E8A33D'


def _iso_dir(session_dir):
    return Path(session_dir) / 'isolation'


def load_cluster(session_dir, ks_dir, clu_id, ks_cache=None):
    """Everything the six columns need for one cluster.

    `session_dir` holds curated.json and the screen's isolation/ output; the
    SortData files are NOT beside the Kilosort data, which is where
    sortdata_path() points by default.
    """
    ks_cache = {} if ks_cache is None else ks_cache
    if ks_dir not in ks_cache:
        ks_cache[ks_dir] = load_ks_dir(ks_dir, exclude_noise=False)
    sp = ks_cache[ks_dir]

    idx = np.where(sp.clu == clu_id)[0]
    out = {'clu_id': clu_id, 'n_spikes': int(idx.size),
           'times': sp.st[idx], 'amps': sp.temp_scaling_amps[idx],
           'feats': None, 'wf': None, 'noise_wf': None}

    if sp.pc_feat is not None and idx.size:
        npc = min(N_PC, sp.pc_feat.shape[1])
        flat = sp.pc_feat[idx, :npc, :].reshape(idx.size, -1)
        out['feats'] = flat[np.all(np.isfinite(flat), axis=1)]

    # Waveforms: pool the per-frame means the screen already stored.
    p = _iso_dir(session_dir) / f'{clu_id + 1}.SortData.h5'
    if p.exists():
        import h5py
        with h5py.File(p, 'r') as h:
            cw = [h[f][  'clu_wf'][:] for f in h if 'clu_wf' in h[f]]
            nw = [h[f]['noise_wf'][:] for f in h if 'noise_wf' in h[f]]
        if cw:
            out['wf'] = np.mean(np.vstack(cw), axis=0)
        if nw:
            out['noise_wf'] = np.mean(np.vstack(nw), axis=0)
        # Rule 4a tests the unit against its REFERENCE CHANNEL, not against
        # another cluster, and that comparison lives only here. Without it the
        # biggest rule in the corpus has a firing column with nothing in it.
        with h5py.File(p, 'r') as h:
            u = [h[f]['unit'][:] for f in h if 'unit' in h[f]]
            nz = [h[f]['noise'][:] for f in h if 'noise' in h[f]]
        if u:
            out['screen_unit'] = np.vstack(u)
        if nz:
            out['screen_noise'] = np.vstack(nz)
    return out


def partner_of(row, sp_clu=None):
    """The cluster this one is drawn against, and why.

    Chosen from the record so the CCG and feature-space columns show the
    partner the RULE actually used, not the nearest one by some fresh metric.
    """
    if row.get('redundant_of') is not None:
        return int(row['redundant_of']), 'counted under'
    ov = row.get('overlap_pairs') or []
    if ov:
        p = ov[0]
        return int(p['clu'] if isinstance(p, dict) else p), 'overlapping'
    acc = [a for a in (row.get('ccg_accusers') or [])
           if isinstance(a, dict) and a.get('z') is not None]
    if acc:
        return int(max(acc, key=lambda a: a['z'])['clu']), 'accuser'
    return None, None


def draw_row(axes, rec_row, data, partner=None, partner_why=None,
             show_xlabels=False):
    """Draw one cluster across the six axes. `axes` is length 6."""
    a_wf, a_isi, a_prj, a_amp, a_fs, a_ccg = axes
    fired = rec_row.get('_fires')           # column name, set by the caller

    def frame(ax, col):
        for s in ax.spines.values():
            s.set_linewidth(2.0 if col == fired else 0.6)
            s.set_color(C_FIRE if col == fired else '#B9BEC6')

    # --- waveform: the unit and the reference channel it is tested against
    if data['wf'] is not None:
        t = np.arange(len(data['wf'])) / FS * 1000
        pk = np.max(np.abs(data['wf'])) + 1e-12
        a_wf.plot(t, data['wf'] / pk, color=C_UNIT, lw=1.4)
        if data['noise_wf'] is not None:
            a_wf.plot(t, data['noise_wf'] / pk, color=C_REF, lw=1.0, ls='--')
        a_wf.set_ylim(-1.35, 1.0)
    else:
        a_wf.text(.5, .5, 'no waveform', ha='center', va='center',
                  transform=a_wf.transAxes, fontsize=6, color=C_QUIET)
    frame(a_wf, 'waveform')

    # --- ISI: the refractory zone is the test, so shade it
    st = data['times']
    if st.size > 2:
        isi = np.diff(np.sort(st)) * 1000
        isi = isi[isi > 0]
        if isi.size:
            bins = np.logspace(-1, np.log10(max(isi.max(), 10)), 40)
            a_isi.hist(isi, bins=bins, color=C_QUIET, lw=0)
            a_isi.set_xscale('log')
            a_isi.axvspan(0.1, 1.0, color=C_FIRE, alpha=.13, lw=0)
    frame(a_isi, 'isi')

    # --- 1-D projection: what the dip test actually sees
    basis = rec_row.get('dip_basis')
    res = None
    if data['feats'] is not None and len(data['feats']) >= 240:
        try:
            res = overmerge_verdict(data['feats'])
        except Exception:
            res = None
    if res and res.get('proj') is not None:
        proj = np.asarray(res['proj'], float)
        dp_rec = rec_row.get('_dip_pval')
        split_is_real = (isinstance(dp_rec, (int, float))
                         and math.isfinite(dp_rec) and dp_rec < DIP_PVAL_THRESH)
        # THE TWO-GROUP COLOURING IS ALWAYS AVAILABLE AND ALMOST ALWAYS WRONG.
        # The split is imposed on every cluster whether or not it has two
        # modes, so colouring it on a unimodal cluster shows a reader two
        # populations that the test just said were not there. Only the dip
        # decides, so only a significant dip gets two colours.
        lab = res.get('labels') if split_is_real else None
        if lab is not None and len(lab) == len(proj) and len(set(lab)) > 1:
            bins = np.linspace(proj.min(), proj.max(), 34)
            for g, col in ((0, C_UNIT), (1, C_G2)):
                a_prj.hist(proj[np.asarray(lab) == g], bins=bins, color=col, lw=0)
        else:
            a_prj.hist(proj, bins=30, color=C_UNIT, lw=0)
        dp = dp_rec                          # from the RECORD, never recomputed
        if isinstance(dp, (int, float)) and math.isfinite(dp):
            a_prj.text(.03, .93, f'dip p = {dp:.3f}', transform=a_prj.transAxes,
                       fontsize=6, va='top',
                       color=C_FIRE if dp < DIP_PVAL_THRESH else C_QUIET)
        if basis:
            a_prj.text(.97, .93, basis, transform=a_prj.transAxes,
                       fontsize=5, color=C_QUIET, ha='right', va='top')
    else:
        a_prj.text(.5, .5, 'dip not evaluated', ha='center', va='center',
                   transform=a_prj.transAxes, fontsize=6, color=C_QUIET)
    frame(a_prj, 'projection')

    # --- amplitude over time: where intermittent presence shows
    if st.size:
        a_amp.scatter(st / 60, data['amps'], s=.7, color=C_UNIT,
                      alpha=.45, linewidths=0)
        af = rec_row.get('absent_frame_frac')
        if isinstance(af, (int, float)) and math.isfinite(af) and af > 0:
            a_amp.text(.03, .93, f'absent {af:.0%}', transform=a_amp.transAxes,
                       fontsize=6, va='top',
                       color=C_FIRE if af > ABSENT_FRAME_FRAC_THRESH else C_QUIET)
    frame(a_amp, 'amplitude')

    # --- feature space: the unit against whatever it was compared with
    if data['feats'] is not None and len(data['feats']):
        f = data['feats']
        a_fs.scatter(f[:, 0], f[:, 1], s=.7, color=C_UNIT, alpha=.4, linewidths=0)
        if partner is not None and partner.get('feats') is not None:
            g = partner['feats']
            a_fs.scatter(g[:, 0], g[:, 1], s=.7, color=C_PARTNER,
                         alpha=.4, linewidths=0)
            a_fs.text(.03, .93, f'vs clu {partner["clu_id"]}',
                      transform=a_fs.transAxes, fontsize=6, va='top',
                      color=C_PARTNER)
        elif data.get('screen_unit') is not None and data.get('screen_noise') is not None:
            a_fs.clear()
            u, nz = data['screen_unit'], data['screen_noise']
            a_fs.scatter(nz[:, 0], nz[:, 1], s=.7, color=C_REF, alpha=.5,
                         linewidths=0)
            a_fs.scatter(u[:, 0], u[:, 1], s=.7, color=C_UNIT, alpha=.5,
                         linewidths=0)
            a_fs.text(.03, .93, 'vs reference channel',
                      transform=a_fs.transAxes, fontsize=6, va='top',
                      color=C_REF)
    sep = rec_row.get('between_cluster_confirmed_separated')
    if sep is None and data.get('screen_unit') is None:
        a_fs.text(.5, .06, 'not evaluated', ha='center',
                  transform=a_fs.transAxes, fontsize=6, color=C_QUIET)
    frame(a_fs, 'feature_space')

    # --- CCG against that same partner, with the calibrated gate named
    if partner is not None and partner.get('times') is not None and st.size:
        try:
            # compute_ccg takes RAW SAMPLES and divides by fs itself. Handing
            # it seconds compressed a 10-minute recording into 20 ms, so every
            # pair fell inside the window and every CCG came out a triangle.
            # Same window and bin as the rule, so the panel shows what the gate
            # saw rather than a prettier version of it.
            centers, counts = compute_ccg(st * FS, partner['times'] * FS,
                                          window_ms=CCG_WINDOW_MS,
                                          bin_ms=CCG_BIN_MS, fs=FS)
            a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                      color=C_QUIET, lw=0)
            a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.13, lw=0)
            if partner_why:
                a_ccg.text(.03, .93, partner_why, transform=a_ccg.transAxes,
                           fontsize=6, va='top', color=C_QUIET)
        except Exception as e:
            a_ccg.text(.5, .5, f'ccg failed', ha='center', va='center',
                       transform=a_ccg.transAxes, fontsize=6, color=C_QUIET)
    elif st.size > 2:
        # No cluster partner: show this unit against ITSELF. An empty panel
        # says nothing; the autocorrelogram at least shows the refractory
        # structure, and is labelled so it is not mistaken for a CCG.
        centers, counts = compute_ccg(st * FS, st * FS,
                                      window_ms=CCG_WINDOW_MS,
                                      bin_ms=CCG_BIN_MS, fs=FS)
        a_ccg.bar(centers, counts, width=(centers[1] - centers[0]),
                  color=C_QUIET, lw=0)
        a_ccg.axvspan(-1, 1, color=C_FIRE, alpha=.13, lw=0)
        a_ccg.text(.03, .93, 'autocorrelogram', transform=a_ccg.transAxes,
                   fontsize=6, va='top', color=C_QUIET)
    else:
        a_ccg.text(.5, .5, 'no partner', ha='center', va='center',
                   transform=a_ccg.transAxes, fontsize=6, color=C_QUIET)
    frame(a_ccg, 'ccg')

    for ax in axes:
        ax.tick_params(labelsize=5, length=2, pad=1)
        if not show_xlabels:
            ax.set_xticklabels([])
        ax.set_yticks([])
