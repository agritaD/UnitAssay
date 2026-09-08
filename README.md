# unitassay

**Measure, curate and link spike-sorted units — and say which ones it cannot decide.**

`unitassay` runs after a spike sorter. It scores how well each cluster is
isolated, decides which clusters are usable, and names *what is wrong* with the
ones that are not — over-split, duplicate, contaminated — rather than returning
a bare keep/reject or a table of metrics.

It also says when it does not know. That is the point: on a synthetic corpus
with ground truth it reaches **83.7% balanced accuracy**, not 100%, and the
clusters it cannot adjudicate are labelled as such instead of guessed.

---

## Install

```bash
pip install unitassay          # once released
pip install -e .               # from a clone
```

Python ≥ 3.10. No GPU. The sorter runs before this and is not included.

Two optional extras, neither needed to curate:

```bash
pip install 'unitassay[sort]'  # kilosort + torch, to run the sorter itself
pip install 'unitassay[si]'    # spikeinterface, for its quality metrics
```

## Use

One recording:

```bash
unitassay run --ks-dir /path/to/kilosort_output --out /path/to/results
```

Many, with one output subdirectory each:

```bash
unitassay batch --sessions sessions.tsv --out /path/to/results --n-jobs 6
```

`sessions.tsv` is `name<TAB>ks_dir`, one recording per line, `#` for comments.
The corpus manifest is an input rather than something the tool knows, so it runs
on your recordings without being told about them in code.

A batch resumes, and the resume will not serve a stale result: a stored
`curated.json` is trusted only if it records **this** version of unitassay and
its isolation files are still beside it. Anything else is recomputed.

```python
from unitassay import curate

rows = curate.curate_session(ks4_dir, isolation_dir='isolation')
clean = [r for r in rows if r['outcome'] == 'clean']
```

Input is a standard Phy-style sorter directory: `spike_times.npy`,
`spike_clusters.npy`, `pc_features.npy`, `pc_feature_ind.npy`, `templates.npy`,
`channel_positions.npy`, `params.py`.

---

## The three layers

| layer | what it does | status |
|---|---|---|
| `isolate` | per-frame L-ratio of each unit against a shape-dissimilar reference channel and against each neighbouring cluster | ✅ |
| `curate` | eight rules that decide keep/reject, a taxonomy naming the mechanism, and a findings layer naming the error class | ✅ |
| `link` | matching the same unit across recordings | in progress |

## What a row tells you

Three vocabularies, answering different questions. They are kept apart on
purpose — collapsing them is how a tool ends up asserting more than it measured.

```python
{'decision': 'FAIL',                    # keep or reject
 'rule': '4', 'tag': 'confirmed_fail_within_cluster',
 'sub_type': 'within_cluster_contamination',   # the MECHANISM — which test fired
 'outcome': 'oversplit',                       # the ERROR CLASS — what the cluster IS
 'findings': ['contaminated', 'refractory'],   # EVERY finding, not just the first
 'not_run': []}                                # tests that could not run
```

`sub_type` names the test. `outcome` names the cluster. The map between them is
measured against ground truth, never assumed — `within_cluster_contamination`
turns out to be over-split 79% of the time, and the tool says so rather than
implying it.

### The rules, in the order they are checked

| | rule | asks | blind to |
|---|---|---|---|
| 1 | identity | already counted under another cluster? | nothing |
| 2 | refractory | can one neuron have produced these spikes? | other clusters |
| 3 | composition | two populations inside it? | other clusters, spike time |
| 4 | contamination | polluted by something this cluster alone can see? | **other clusters** |
| 5 | neighbour | separable from the neighbours it is tested against? | unflagged neighbours |
| 6 | unresolved | could it be tested at all? | everything else |
| 7 | clean | nothing found | — |
| 8 | fall-through | failed upstream, nothing attributed it | why |

Each rule is named for **what it can see**, not for what it proves. Rule 4
compares a cluster to a reference *channel* and never looks at another cluster,
so it can establish that foreign events are present but not who they belong to.
Only rule 5 can, and only for neighbours that were flagged.

## Outcomes

| outcome | meaning |
|---|---|
| `clean` | nothing fired |
| `duplicate` | these spikes are already counted under another cluster |
| `oversplit` | a fragment of a neuron that lives elsewhere |
| `mixed_cluster` | two populations, cause **not resolved** |
| `intermittent_presence` | absent for much of the recording, clean where present |
| `low_spike_count` | below the floor at which any test can run |

`mixed_cluster` makes no claim about the error class. Three independent tests
were tried on it — feature-space bimodality, a within-cluster cross-correlogram,
and the auto-correlogram — and none separates "two neurons merged" from "one
fragment that is itself bimodal" at realistic spike counts. Reporting it as
over-merge would be wrong about two thirds of the time.

## Performance

Benchmarked on a synthetic corpus with ground truth (28 recordings, 3,696
clusters ≥ 240 spikes, 30.8% clean):

| tool | sensitivity | specificity | balanced |
|---|---|---|---|
| sorter label | 84.5% | 59.5% | 72.0% |
| three fixed thresholds | 83.2% | 58.7% | 70.9% |
| isolation alone | 76.8% | 73.2% | 75.0% |
| **unitassay** | **81.3%** | **86.1%** | **83.7%** |

Precision and F1 are deliberately absent: both move with the corpus's clean
rate, so they describe the recording as much as the tool. Sensitivity,
specificity and balanced accuracy are prevalence-independent and transfer.

**Cost**, 147 real sessions, 53,428 clusters, run from scratch:
114 ms per cluster to isolate, 89 ms to curate — 32 minutes wall clock on 6
workers.

## Limitations

- **Over-merge is not separable from over-split** in a single cluster's spikes
  and features at realistic counts. Detecting a 50% refractory dip between two
  sub-populations needs ~1,900 spikes in each; the observed median is 319.
- **No noise-contamination detector.** Contaminated clusters are rejected, but
  attributed to another cause: the within-cluster test cannot tell noise from a
  neighbour, because it never looks at a neighbour.
- **Drift is tagged, not corrected.** Under drift the false-positive rate rises
  roughly fourfold; those clusters carry a `high_drift` overlay.
- **Ground truth is synthetic.** Real recordings have no ground truth, so the
  147-session numbers are counts and proportions, never accuracies.

## Authors

Developed by **Agrita Dubey** and **Qiao (Betty) Xu**, Pesaran Lab, University
of Pennsylvania. Developed with Claude (Anthropic).

`isolate` was previously developed under the internal name `mind_snag`; the
pipeline, config keys and file formats are unchanged.

## Licence

MIT — see [LICENSE](LICENSE).
