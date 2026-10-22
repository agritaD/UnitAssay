#!/usr/bin/env python3
"""Build the tiny synthetic sorter output that ships with unitassay.

WHY A FIXTURE EXISTS AT ALL. Without one, nobody who does not already have
Neuropixels recordings can run this tool -- not a reviewer, not a new user, not
CI -- and the README's example is unrunnable text. `tests/fixture/` is a
complete Phy-style directory: four clusters, eight channels, 300 s.

WHAT IT IS AND IS NOT. It is SYNTHETIC and deterministic (seed 0), built here
from scratch rather than cut from a recording, so it carries no data-sharing
question and no subject's data. It is NOT a benchmark: four hand-made clusters
say nothing about accuracy, which is measured on the synthetic corpus with
ground truth. Its job is to prove the pipeline RUNS end to end and to catch a
change that silently alters a verdict.

THE FOUR CLUSTERS are built to land in different places, so the golden file is
not four copies of one answer:
    0  a clean, well-isolated unit on channel 1
    1  a second clean unit on channel 5, far away
    2  a refractory violator -- 12% of its spikes within 1.5 ms of the last
    3  under the 240-spike floor, where no test can run

Regenerate with:  python tests/make_fixture.py
The output is committed; regenerating must not change it unless intended.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""
import sys
from pathlib import Path

import numpy as np

FS = 30000.0
DUR_S = 300.0
N_CH = 8
N_PC_CH = 8              # channels kept per spike in pc_features
N_PC = 3                 # principal components per channel
N_SAMP = 61              # template length
OUT = Path(__file__).parent / 'fixture'

# (n_spikes, peak channel, refractory-violation fraction)
CLUSTERS = [(700, 1, 0.00),
            (650, 5, 0.00),
            (600, 3, 0.12),
            (120, 6, 0.00)]


def _spike_times(rng, n, viol_frac):
    """Poisson-ish times, optionally with a burst of refractory violations."""
    t = np.sort(rng.uniform(0, DUR_S, n))
    n_viol = int(round(n * viol_frac))
    if n_viol:
        # place violations 0.3-1.5 ms after an existing spike, which is inside
        # the 1.5 ms refractory window the ISI test uses
        idx = rng.choice(n - n_viol, n_viol, replace=False)
        extra = t[idx] + rng.uniform(0.0003, 0.0015, n_viol)
        t = np.sort(np.concatenate([t[:n - n_viol], extra]))
    return t


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    times, clusters, amps, feats = [], [], [], []
    for cid, (n, peak, viol) in enumerate(CLUSTERS):
        t = _spike_times(rng, n, viol)
        times.append(t)
        clusters.append(np.full(t.size, cid))
        amps.append(rng.normal(40 + 5 * cid, 3.0, t.size))
        # PC space: a compact cloud centred away from the origin on the peak
        # channel, with the other channels near zero. Two clusters at different
        # peaks are therefore separable, which is what the L-ratio measures.
        f = rng.normal(0, 1.0, (t.size, N_PC, N_PC_CH))
        f[:, 0, peak] += 12.0
        f[:, 1, peak] += 4.0
        feats.append(f)

    order = np.argsort(np.concatenate(times))
    st = (np.concatenate(times)[order] * FS).astype(np.int64)
    cl = np.concatenate(clusters)[order].astype(np.int32)
    am = np.concatenate(amps)[order].astype(np.float64)
    pf = np.concatenate(feats, axis=0)[order].astype(np.float32)

    # one template per cluster: a negative-going spike on its peak channel,
    # decaying with distance so `noise_channel` (min-energy local channel) has
    # a well-defined answer
    tmpl = np.zeros((len(CLUSTERS), N_SAMP, N_CH), dtype=np.float64)
    tt = np.arange(N_SAMP)
    shape = -np.exp(-((tt - 20) ** 2) / 18.0) + 0.35 * np.exp(-((tt - 32) ** 2) / 40.0)
    for cid, (_n, peak, _v) in enumerate(CLUSTERS):
        for ch in range(N_CH):
            tmpl[cid, :, ch] = shape * np.exp(-abs(ch - peak) / 1.2) * 50.0

    pos = np.zeros((N_CH, 2))
    pos[:, 0] = np.where(np.arange(N_CH) % 2, 16.0, 0.0)      # two columns
    pos[:, 1] = (np.arange(N_CH) // 2) * 20.0                 # 20 um rows

    np.save(OUT / 'spike_times.npy', st)
    np.save(OUT / 'spike_clusters.npy', cl)
    np.save(OUT / 'spike_templates.npy', cl.astype(np.int32))
    np.save(OUT / 'amplitudes.npy', am)
    np.save(OUT / 'pc_features.npy', pf)
    np.save(OUT / 'pc_feature_ind.npy',
            np.tile(np.arange(N_PC_CH, dtype=np.int32), (len(CLUSTERS), 1)))
    np.save(OUT / 'templates.npy', tmpl)
    np.save(OUT / 'channel_positions.npy', pos)
    np.save(OUT / 'channel_map.npy', np.arange(N_CH, dtype=np.int32))
    np.save(OUT / 'whitening_mat_inv.npy', np.eye(N_CH))
    # KSLabel is the sorter's own verdict, which the tool overrides but reads:
    # cluster 2 is the refractory violator, so KS calls it mua. Giving every
    # cluster the same label would make the screen-vs-curate contrast
    # vacuous on this fixture.
    (OUT / 'cluster_KSLabel.tsv').write_text(
        'cluster_id\tKSLabel\n'
        + ''.join(f'{cid}\t{"mua" if cid == 2 else "good"}\n'
                  for cid in range(len(CLUSTERS))))
    (OUT / 'cluster_group.tsv').write_text(
        'cluster_id\tgroup\n'
        + ''.join(f'{cid}\t{"mua" if cid == 2 else "good"}\n'
                  for cid in range(len(CLUSTERS))))
    (OUT / 'params.py').write_text(
        "dat_path = 'not_needed.bin'\n"       # no raw data: this is post-sorter
        "n_channels_dat = %d\n"
        "dtype = 'int16'\n"
        "offset = 0\n"
        "sample_rate = %r\n"
        "hp_filtered = True\n" % (N_CH, FS))

    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f'{OUT}: {len(CLUSTERS)} clusters, {st.size} spikes, '
          f'{N_CH} channels, {DUR_S:.0f} s, {total / 1024:.0f} KB')
    for cid, (n, peak, viol) in enumerate(CLUSTERS):
        print(f'  cluster {cid}: {int((cl == cid).sum()):>4} spikes  '
              f'peak ch {peak}  refractory violations {viol:.0%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
