#!/usr/bin/env python3
"""curate/sortdata.py

Canonical reader for mind_snag's per-cluster SortData.h5 files (v2 HDF5
format: frame_0000, frame_0001, ... groups, each with unit/noise/other
PC-feature datasets), plus the noise-channel identification helper.
Consolidated 2026-07-13 from ~4 independent readers of the same format
(_load_ms_frames, _load_ms_worst_frame, _load_ms_v2 family, and
process_sortdata) scattered across noise-fail / neighbor-fail scripts.

INDEXING CONVENTION (this is the ONE place this conversion happens):
  - clu_id is ALWAYS 0-indexed, matching spike_clusters.npy / templates.npy
    / pc_feature_ind.npy / amplitudes.npy (KS4-native).
  - ms_id is ALWAYS 1-indexed = clu_id + 1, matching mind_snag's
    "{ms_id}.SortData.h5" filename convention.
  - Every public function here takes clu_id (0-indexed) as its parameter
    and does the +1 internally, exactly once, right where the filename is
    built. Callers should never compute ms_id themselves -- that repeated
    "+1 in N different places" pattern is exactly what produced the
    indexing bug this module is meant to prevent.
  - SortData.h5's own 'other_clu' field stores NEIGHBOR ids 1-indexed
    (ms-native); load_ms_frames() converts these back to 0-indexed clu_ids
    before returning them, so everything this module hands back to a
    caller is 0-indexed KS4-native, with no exceptions.

Does NOT cover the older v1 (MATLAB HDF5 object-ref cell array) SortData
format -- that's a different on-disk schema, not just an indexing
difference, and stays in its own reader (_load_ms_v1 family) until/unless
someone confirms no v1-format sessions still need active analysis.

Author: Agrita Dubey
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

import numpy as np
from unitassay.curate.constants import L_RATIO_MS_THRESH, ISOLATED_T_RATIO
import h5py
from pathlib import Path

from unitassay.curate.lratio import ms_l_ratio


def sortdata_path(ks4_path, clu_id, isolation_dir='isolation'):
    """Path to clu_id's (0-indexed) SortData.h5 file. The ONE place ms_id
    = clu_id + 1 is computed for building this specific filename."""
    ms_id = clu_id + 1
    return Path(ks4_path) / isolation_dir / f'{ms_id}.SortData.h5'


def noise_channel(templates, pc_feat_ind, clu_id):
    """Min-energy local channel for clu_id (0-indexed) -- mind_snag's
    definition of a cluster's "noise channel" (weakest local signal)."""
    local_chs = pc_feat_ind[clu_id]
    tmpl_local = templates[clu_id][:, local_chs]
    energies = np.sum(tmpl_local ** 2, axis=0)
    return int(local_chs[np.argmin(energies)])


def load_ms_frames(ks4_path, clu_id, isolation_dir='isolation'):
    """Load all frames of clu_id's (0-indexed) SortData.h5, computing
    per-frame noise/neighbor L-ratios (curate.lratio.ms_l_ratio, unit as
    reference) along the way.

    Returns None if the file doesn't exist or can't be read. Otherwise a
    dict:
      n_frames       : int
      other_clu_ids  : list[int], 0-indexed KS4 cluster ids of neighbors
                       sharing clu_id's peak channel (converted from
                       SortData's 1-indexed 'other_clu' field)
      lr_noise       : (n_frames,) array, ms_l_ratio(unit, noise) per frame
      lr_others      : (n_others, n_frames) array, ms_l_ratio(unit, other_j)
                       per frame per neighbor
      frames         : list of length n_frames, each either None (frame
                       missing / too few unit spikes) or a dict with keys
                       'unit', 'noise' (each (n_spikes, n_pc) or None),
                       'others' (list of (n_spikes, n_pc) arrays, one per
                       neighbor, len == len(other_clu_ids))
    """
    sf = sortdata_path(ks4_path, clu_id, isolation_dir)
    if not sf.exists():
        return None
    try:
        with h5py.File(sf, 'r') as f:
            n_frames = int(f.attrs.get('n_frames', 0))
            other_clu_ids = []
            if 'frame_0000' in f and 'other_clu' in f['frame_0000']:
                other_clu_ids = [int(x) - 1 for x in f['frame_0000']['other_clu'][:]]
            n_others = len(other_clu_ids)

            lr_noise_arr = np.full(n_frames, np.nan)
            lr_others_arr = np.full((n_others, n_frames), np.nan)
            frames = []

            for i in range(n_frames):
                fk = f'frame_{i:04d}'
                if fk not in f:
                    frames.append(None)
                    continue
                fr = f[fk]
                unit = np.array(fr['unit']).astype(float) if 'unit' in fr else None
                noise = np.array(fr['noise']).astype(float) if 'noise' in fr else None
                if unit is None or unit.ndim < 2 or unit.shape[0] <= unit.shape[1]:
                    frames.append(None)
                    continue

                lr_noise_arr[i] = (ms_l_ratio(unit, noise)
                                   if (noise is not None and noise.ndim == 2 and noise.shape[0] > 0)
                                   else 0.0)

                others_pc = []
                if 'other' in fr and hasattr(fr['other'], 'keys'):
                    for k in sorted(fr['other'].keys()):
                        others_pc.append(np.array(fr['other'][k]).astype(float))
                while len(others_pc) < n_others:
                    others_pc.append(np.zeros((0, 3)))
                for j in range(n_others):
                    ot = others_pc[j]
                    lr_others_arr[j, i] = ms_l_ratio(unit, ot) if ot.shape[0] > 0 else 0.0

                frames.append({'unit': unit, 'noise': noise, 'others': others_pc[:n_others]})

            return {
                'n_frames': n_frames,
                'other_clu_ids': other_clu_ids,
                'lr_noise': lr_noise_arr,
                'lr_others': lr_others_arr,
                'frames': frames,
            }
    except Exception:
        return None


def worst_frame(result):
    """Given load_ms_frames()'s return dict, pick the frame with the
    highest max-over-neighbors L-ratio (mind_snag's own "worst frame" for
    feature-space visualization), falling back to the first valid frame.
    Returns None if there's no valid frame at all."""
    n_others = result['lr_others'].shape[0]
    lr_other_max = (np.nanmax(result['lr_others'], axis=0) if n_others > 0
                    else np.zeros(result['n_frames']))
    idx = int(np.nanargmax(lr_other_max)) if np.any(np.isfinite(lr_other_max)) else 0
    fd = result['frames'][idx] if idx < len(result['frames']) else None
    if fd is None:
        for fd in result['frames']:
            if fd is not None:
                break
    return fd


# ---------------------------------------------------------------------------
# MOVED HERE FROM analysis/ 2026-08-30. Both are pure functions over a
# SortData.h5 and were living in analysis scripts, which meant curate/curation.py
# -- the toolbox entry point -- had to import from analysis/ to assemble
# curate()'s input. A toolbox that depends on its own callers is not standalone,
# and it is also how six copies of the pipeline came to exist. The original
# sites re-export these names so existing imports keep working.
# ---------------------------------------------------------------------------


def process_sortdata_all_frames(sf_path):
    try:
        with h5py.File(sf_path, 'r') as f:
            n_frames = int(f.attrs.get('n_frames', 0))
            if n_frames == 0:
                return None
            other_clu_ids = []
            if 'frame_0000' in f and 'other_clu' in f['frame_0000']:
                other_clu_ids = [int(x) - 1 for x in f['frame_0000']['other_clu'][:]]
            n_others = len(other_clu_ids)
            lr_noise = np.full(n_frames, np.nan)
            lr_others = [np.full(n_frames, np.nan) for _ in range(n_others)]
            for i in range(n_frames):
                fk = f'frame_{i:04d}'
                if fk not in f:
                    continue
                fr = f[fk]
                unit = np.array(fr['unit']).astype(np.float64) if 'unit' in fr else None
                if unit is None or unit.ndim < 2 or unit.shape[0] < 4:
                    continue
                if 'noise' in fr:
                    noise = np.array(fr['noise']).astype(np.float64)
                    lr_noise[i] = ms_l_ratio(unit, noise) if noise.ndim == 2 and noise.shape[0] > 0 else 0.0
                else:
                    lr_noise[i] = 0.0
                if 'other' in fr and hasattr(fr['other'], 'keys'):
                    other_keys = sorted(fr['other'].keys())
                    for j, k in enumerate(other_keys):
                        if j >= n_others:
                            break
                        ot = np.array(fr['other'][k]).astype(np.float64)
                        lr_others[j][i] = ms_l_ratio(unit, ot) if ot.ndim == 2 and ot.shape[0] > 0 else 0.0
                    for j in range(len(other_keys), n_others):
                        lr_others[j][i] = 0.0
                else:
                    for j in range(n_others):
                        lr_others[j][i] = 0.0
        return {'n_frames': n_frames, 'lr_noise': lr_noise, 'lr_others': lr_others}
    except Exception:
        return None


def channel_fields(sd):
    """mind_snag's per-frame rules, plus the frame COVERAGE the taxonomy needs.

    `absent_frame_frac` and `clean_frac_when_present` are computed here, from the
    `mat`/`absent` this function already builds, because computing them anywhere
    else would be a second implementation of the same matrix.

    WHY THEY MATTER. A NaN in the noise row means the unit had NO SPIKES in that
    frame -- it went silent, drifted out of range, or KS4 stopped matching its
    template. mind_snag counts such a frame AGAINST the clean fraction ("no
    spikes = drifted away = not isolated"), so a cluster absent from more than
    1 - ISOLATED_T_RATIO of frames CANNOT reach the threshold however clean its
    present frames are. That is a COVERAGE failure, not a contamination one, and
    without these two numbers the two are indistinguishable downstream: the
    cluster fails, every attribution channel comes back clean, and
    `taxonomy.classify()` can only call it `unexplained`.

    With them, `taxonomy` re-labels it `intermittent_presence` on its own
    evidence gate (absent > ABSENT_FRAME_FRAC_THRESH AND clean-where-present >=
    ISOLATED_T_RATIO). Note this changes NO decision -- the cluster still fails.
    It changes an admission of ignorance into a stated disagreement with
    `presence_ratio`, which is the whole point for behaviour-modulated units.

    They were already computed for the 50 real sessions in
    m260813_lratio_channel_all.py and nowhere on the synthetic side, so
    `intermittent_presence` fired on 41 real clusters and 0 synthetic ones --
    not because the synthetic corpus lacks intermittent units (it has a
    recording built of them) but because nobody measured this there.
    """
    if sd is None or sd['n_frames'] == 0:
        return None
    lo = (np.stack(sd['lr_others'], 0) if sd['lr_others']
          else np.zeros((0, sd['n_frames'])))
    noise = sd['lr_noise']
    mat = np.vstack([noise[None, :], lo])
    absent = np.isnan(mat[0, :])
    for k in range(1, mat.shape[0]):
        mat[k, np.isnan(mat[k, :]) & (~absent)] = 0.0
    clean = np.all(mat < L_RATIO_MS_THRESH, axis=0)
    other_max = np.nanmax(lo, axis=0) if lo.shape[0] else np.zeros(sd['n_frames'])
    mn_n, mn_o = np.nanmean(noise), np.nanmean(other_max)
    mx_n, mx_o = np.nanmax(noise), np.nanmax(other_max)
    if mn_n >= mn_o and mx_n >= L_RATIO_MS_THRESH:
        dom = 'noise'
    elif mn_o > mn_n and mx_o >= L_RATIO_MS_THRESH:
        dom = 'other_cluster'
    elif mx_n >= L_RATIO_MS_THRESH or mx_o >= L_RATIO_MS_THRESH:
        dom = 'mixed'
    else:
        dom = 'lratio_passed'
    present = ~absent
    return dict(passes_lratio=bool(clean.sum() / sd['n_frames'] >= ISOLATED_T_RATIO),
                dominant_source=dom,
                neighbor_never_fails=bool(mx_o < L_RATIO_MS_THRESH),
                absent_frame_frac=float(absent.mean()),
                # "would it pass if judged ONLY where the unit is present?"
                # 0.0 when it is present in no frame at all -- that is a real
                # verdict (nothing clean was observed), not a missing value.
                clean_frac_when_present=(float(np.all(
                    mat[:, present] < L_RATIO_MS_THRESH, axis=0).mean())
                    if present.any() else 0.0))


# mind_snag's own aggregation constants, restated here because the two that
# matter are not in curate/constants: L_RATIO_MS_THRESH is (auto_curate.py), and
# MS_FRAME_FRAC is 1 - ISOLATED_T_RATIO -- failing more than that many frames is
# what sinks a cluster.
MS_FRAME_FRAC = 1.0 - ISOLATED_T_RATIO


def flagged_neighbours(sf_path):
    """Every neighbour that ACCUSES this cluster, not just the argmax.

    Moved into curate 2026-09-04 (task A11) from
    analysis/m260821_rule5_scope_all_neighbours.py so `curate_session` can build
    rule 5's inputs itself. Behaviour is unchanged.

    WHAT COUNTS AS AN ACCUSER matters more than it looks, and the two obvious
    choices disagree by a factor of three:

      mean over frames >= 0.2   the aggregation that picks the DOMINANT
                                neighbour. Strict: a neighbour that fails half
                                the frames badly and the rest cleanly can
                                average under threshold.
      crosses in >= 40% of frames
                                mind_snag's OWN aggregation -- failing more than
                                1 - ISOLATED_T_RATIO of frames is what sinks a
                                cluster.

    The UNION of the two is used. mind_snag's is the criterion that actually
    condemns a cluster, and the mean criterion is what selected the one
    neighbour originally tested, so dropping either would understate the scope
    or lose the dominant neighbour itself. A third, looser choice (crossed in
    ANY frame) finds 71% of retrievals with >1 accuser against the mean
    criterion's 25%.

    Returns (flagged, all_stats), both sorted by mean_lr descending, so the
    dominant neighbour is `flagged[0]` whenever it qualifies at all. `all_stats`
    is kept so a cluster with NO qualifying accuser can still be asked why:
    `dominant_source == 'other_cluster'` with nothing crossing either threshold
    puts the attribution itself in question, which is a different finding from
    the scope gap. Each entry carries `criterion`, so the strict mean-only
    subset stays recoverable by filtering.
    """
    sd = process_sortdata_all_frames(sf_path)
    if sd is None or not sd['lr_others']:
        return None, None
    with h5py.File(sf_path, 'r') as f:
        if 'frame_0000' not in f or 'other_clu' not in f['frame_0000']:
            return None, None
        other_clu_ids = [int(x) - 1 for x in f['frame_0000']['other_clu'][:]]
    if not other_clu_ids or len(other_clu_ids) != len(sd['lr_others']):
        return None, None
    out, allst = [], []
    for j, lr in enumerate(sd['lr_others']):
        lr = np.asarray(lr, float)
        if np.all(np.isnan(lr)):
            continue
        m = float(np.nanmean(lr))
        frac = float(np.nanmean(lr >= L_RATIO_MS_THRESH))
        by_mean, by_frames = m >= L_RATIO_MS_THRESH, frac >= MS_FRAME_FRAC
        ent = dict(clu_id=other_clu_ids[j], mean_lr=m, frac_frames_over=frac,
                   criterion='both' if (by_mean and by_frames)
                             else ('mean' if by_mean else
                                   ('frames' if by_frames else None)))
        allst.append(ent)
        if by_mean or by_frames:
            out.append(ent)
    out.sort(key=lambda d: -d['mean_lr'])
    allst.sort(key=lambda d: -d['mean_lr'])
    return out, allst


def aligned_features(pc_feat, pc_feat_ind, clu_a, clu_b, spike_clu, n_pc=3):
    """PC features for two clusters on the channels they SHARE.

    Moved into curate 2026-09-04 (task A11) from
    analysis/m260729_between_cluster_real_check.py. Behaviour unchanged.

    The shared-channel restriction is what makes the two feature sets
    comparable at all: KS4 stores each cluster's features on its own local
    channel set, so the same column index means a different channel for
    different clusters. Returns (fa, fb, n_shared); n_shared < 2 returns
    (None, None, 0) because a one-channel comparison cannot separate anything.
    """
    ind_a, ind_b = pc_feat_ind[clu_a], pc_feat_ind[clu_b]
    shared = np.intersect1d(ind_a, ind_b)
    if len(shared) < 2:
        return None, None, 0
    cols_a = np.array([np.where(ind_a == ch)[0][0] for ch in shared])
    cols_b = np.array([np.where(ind_b == ch)[0][0] for ch in shared])
    n_avail = min(n_pc, pc_feat.shape[1])
    idx_a = np.where(spike_clu == clu_a)[0]
    idx_b = np.where(spike_clu == clu_b)[0]
    fa = pc_feat[idx_a][:, :n_avail, :][:, :, cols_a].reshape(len(idx_a), -1)
    fb = pc_feat[idx_b][:, :n_avail, :][:, :, cols_b].reshape(len(idx_b), -1)
    fa = fa[np.all(np.isfinite(fa), axis=1)]
    fb = fb[np.all(np.isfinite(fb), axis=1)]
    return fa, fb, int(len(shared))
