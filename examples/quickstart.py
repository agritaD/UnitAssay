"""UnitAssay quickstart: curate the two example recordings.

    python examples/quickstart.py --data path/to/unitassay_example_data

Download the example data first (see examples/README.md). It holds two
Kilosort 4 outputs, about 115 MB together:

  real/       2.9 min of macaque posterior parietal cortex, Neuropixels 1.0
  synthetic/  10 min MEArec simulation of 102 neurons in close pairs, with
              ground truth

The script runs UnitAssay on each (under a minute apiece on 8 threads) and shows
how to read the result:

  1. decisions:     how many clusters PASS, FAIL or are too small to test
  2. why:           which rule decided each FAIL, and the error class it names
  3. pairs:         duplicate and over-split findings, which belong to a pair
  4. drift:         how far the probe moved, from Kilosort's own estimate
  5. ground truth:  (synthetic only) is a kept cluster really one clean neuron?

Each step's result is checked against the reference run shipped with the data
(expected_curated.tsv), so a mismatch means your install differs from the one
behind the paper.

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from unitassay.cli import main as unitassay_cli
from unitassay.drift import exclude_gap_blocks


def read_tsv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def curate_one(name, ks_dir, out):
    print(f'\n=== {name}: {ks_dir}')
    # the same as `unitassay run --ks-dir ... --out ...` on the command line
    unitassay_cli(['run', '--ks-dir', str(ks_dir), '--out', str(out)])
    rows = json.loads((out / 'curated.json').read_text())['rows']
    pj = json.loads((out / 'pairs.json').read_text())
    pairs, ps = pj['pairs'], pj['summary']

    # 1. decisions. Count from `decision`; it is always one of these three.
    dec = Counter(r['decision'] for r in rows)
    print(f"\n1. decisions   PASS {dec['PASS']}   FAIL {dec['FAIL']}   "
          f"LOW_SPIKE_COUNT {dec['LOW_SPIKE_COUNT']} (too few spikes to test)")

    # 2. why. `rule_label` names the rule that decided; `error_class` names what
    # is wrong (oversplit, duplicate, contaminated, ...).
    fails = [r for r in rows if r['decision'] == 'FAIL']
    print('\n2. what decided the FAILs')
    for lab, n in Counter(r['rule_label'] for r in fails).most_common():
        print(f'   {n:4d}  {lab}')
    print('   error classes over all clusters:',
          dict(Counter(r['error_class'] for r in rows).most_common()))

    # 3. pairs. A fragment is a fragment OF something, so pairwise errors are
    # reported in pairs.json as well as per cluster.
    print(f"\n3. pairs   {ps['n_pairs']} flagged, {ps['n_not_adjudicated']} "
          f"measured but not adjudicated, {ps['n_both_pass']} with both members PASS")
    for p in [p for p in pairs if p['both_pass']][:3]:
        print(f"   clusters {p['a']} & {p['b']}  [{p['channel']}]  {p['why']}")

    # 4. drift, from Kilosort's ops.npy. Gap blocks (multi-bank probes only)
    # are excluded; a single-bank probe has none.
    d = exclude_gap_blocks(ks_dir)
    print(f"\n4. drift   range_90 {d['range_90_gap_excluded']:.1f} um "
          f"(gap blocks excluded: {d['gap_blocks_excluded']})")

    # check against the reference decisions shipped with the data
    ref = {int(r['clu_id']): r for r in read_tsv(Path(ks_dir).parent /
                                                  'expected_curated.tsv')}
    diff = [r['clu_id'] for r in rows
            if (r['decision'], r['rule_id']) !=
               (ref[r['clu_id']]['decision'], ref[r['clu_id']]['rule_id'])]
    print(f'\n   reference check: {len(rows) - len(diff)}/{len(rows)} clusters '
          f'match expected_curated.tsv' + (f'; differ: {diff[:10]}' if diff else ''))
    return rows


def score(rows, data):
    """5. Score the synthetic run against ground truth.

    A cluster should be KEPT only if it is one neuron's spikes with little else
    (`clean_primary`); fragments, merges and noise clusters should be rejected.
    Clusters below the 240-spike floor, or with no matching neuron, are not
    scored (`evaluated` is False).
    """
    gt = [g for g in read_tsv(data / 'synthetic' / 'ground_truth_clusters.tsv')
          if g['evaluated'] == 'True']
    kept = {r['clu_id'] for r in rows if r['decision'] == 'PASS'}
    keep = [g for g in gt if g['correct_action'] == 'keep']
    rej = [g for g in gt if g['correct_action'] == 'reject']
    sens = sum(int(g['clu_id']) in kept for g in keep) / len(keep)
    spec = sum(int(g['clu_id']) not in kept for g in rej) / len(rej)
    print(f'\n5. ground truth   {len(gt)} clusters scored ({len(keep)} should be '
          f'kept, {len(rej)} rejected)')
    print(f'   sensitivity {sens:.1%}   specificity {spec:.1%}   '
          f'balanced accuracy {(sens + spec) / 2:.1%}')
    for cat, n in Counter(g['gt_category'] for g in rej
                          if int(g['clu_id']) in kept).most_common():
        print(f'   kept but should be rejected: {n} {cat}')
    info = json.loads((data / 'synthetic' / 'recording_info.json').read_text())
    print(f"   reference: balanced accuracy "
          f"{info['expected_score']['balanced_accuracy']:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, type=Path,
                    help='the unzipped unitassay_example_data folder')
    ap.add_argument('--out', default='unitassay_quickstart_out', type=Path,
                    help='where UnitAssay writes its isolation files')
    a = ap.parse_args()
    curate_one('real recording', a.data / 'real' / 'kilosort4', a.out / 'real')
    rows = curate_one('synthetic recording', a.data / 'synthetic' / 'kilosort4',
                      a.out / 'synthetic')
    score(rows, a.data)
    print(f'\nBrowse either result:  unitassay view {a.out / "real"}')


if __name__ == '__main__':
    main()
