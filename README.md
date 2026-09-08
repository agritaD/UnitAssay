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

Seven dependencies, no extras: numpy, scipy, scikit-learn, h5py, diptest,
click, pyyaml.

**The sorter is not included and is not driven from here.** unitassay takes a
sorter's output directory. Kilosort is a separate program (and GPL-3.0, against
this package's MIT); run it, and any preprocessing, from your own pipeline
script.

## Try it without any data

A tiny synthetic sorter directory ships with the package, so the tool is
runnable straight from a clone — four clusters, eight channels, 300 s, about a
second:

```bash
unitassay run --ks-dir tests/fixture --out /tmp/demo
# 4 clusters   PASS 3  FAIL 1
```

One clean unit each on two distant channels, one built with 12% of its spikes
inside the refractory window (fails at rule `2`), and one below the 240-spike
floor (returned as `low_spike_count`). It is a smoke test and a golden
regression, **not a benchmark** — four hand-made clusters say nothing about
accuracy. Rebuild it with `python tests/make_fixture.py`; the seed is fixed and
it regenerates byte-for-byte.

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

## Pairs, not just clusters

`curate()` decides per cluster, but duplication and over-splitting are properties
of a **pair** — a fragment is a fragment *of* something. Measured on 147 real
sessions, every cluster labelled `duplicate` names its partner (382 of 382) while
only 20.7% of those labelled `oversplit` do. A per-cluster output under-reports
pairwise errors by construction, so every run also writes `pairs.json`:

```json
{"a": 214, "b": 231, "channel": "re_detection",
 "mechanism": "shared_spikes_unadjudicated",
 "criterion_met": false,
 "why": "lag below LAG_MIN_MS: same spikes at ~zero lag, which is either
         re-detection or one artifact seen twice -- not adjudicated",
 "evidence": {"overlap": 0.91, "overlap_other": 0.30, "lag_ms": 0.05},
 "decision_a": "PASS", "decision_b": "PASS", "both_pass": true,
 "action": "review_pair"}
```

**No new statistic and no verdict changes** — these are the measurements the tool
already made, reported instead of discarded. The duplicate criterion is a
conjunction (overlap **and** a non-zero lag), and on real data 98.6% of pairs over
the overlap floor fail the lag half. Those are now reported with
`criterion_met: false` and which half failed, rather than omitted: reporting a
measurement and declining to adjudicate it is a different statement from silence.

`both_pass` is the field to filter on. Under every explanation of a high-overlap
pair — one member is a copy, or both are one artifact — both members passing is
wrong.

**Not yet covered:** division over-split, where two clusters avoid each other at
short lag because one neuron cannot fire inside its own refractory period. There
is no detector for it in this tool; `mechanism` is an open vocabulary so that
channel can be added without changing the contract.

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
 'rule_id': '4a',                       # WHICH RULE — read this one
 'rule_label': '4a reference channel',  # and what it tested
 'stage': '4-contamination',            # where in the flow it stopped
 'tag': 'confirmed_fail_within_cluster',
 'sub_type': 'within_cluster_contamination',   # the MECHANISM — which test fired
 'outcome': 'oversplit',                       # the ERROR CLASS — what the cluster IS
 'findings': ['contaminated', 'refractory'],   # EVERY finding, not just the first
 'not_run': [],                                # tests that could not run
 'rule': '4'}                           # INTERNAL KEY — see the note below
```

> **Read `rule_id`, not `rule`.** `rule` is an internal key kept for backwards
> compatibility, and its numbers do **not** match the table below: `rule: '2'`
> is a clean cluster (rule 7), `rule: '0'` is refractory (rule 2), and identity
> is `rule: 'R'`. `rule_id` is the id this document is indexed by.

`sub_type` names the test. `outcome` names the cluster. The map between them is
measured against ground truth, never assumed — `within_cluster_contamination`
turns out to be over-split 79% of the time, and the tool says so rather than
implying it.

### The rules, in the order they are checked

The number is **position in the flow**; the letter is **which test within that
rule**, lettered in firing order. A rule with one test has no letter. This is
the `rule_id` on every row.

| `rule_id` | rule | the test | asks | blind to |
|---|---|---|---|---|
| `1` | IDENTITY | overlap and lag | already counted under another cluster? | nothing |
| `2` | REFRACTORY | refractory ratio | can one neuron have produced these spikes? | other clusters |
| `3a` | COMPOSITION | bimodality | two populations inside it? | other clusters, spike time |
| `3b` | COMPOSITION | time-ordered split | do the two halves separate in TIME? (drift) | other clusters |
| `4a` | CONTAMINATION | reference channel | polluted by something this cluster alone can see? | **other clusters** |
| `4b` | CONTAMINATION | evidence incomplete | — the test could not run | — |
| `5a` | NEIGHBOUR | feature space | separable from the neighbours it is tested against? | unflagged neighbours |
| `5b` | NEIGHBOUR | coincidence excess | do the two fire together more than chance? (re-detection) | — |
| `5c` | NEIGHBOUR | refractory deficit | *less* than chance? (one neuron's own refractory period) | — |
| `5d` | NEIGHBOUR | evidence incomplete | — the test could not run | — |
| `6` | UNRESOLVED | no verdict | could it be tested at all? | everything else |
| `7` | CLEAN | all rules passed | nothing found | — |
| `8` | FALL-THROUGH | unattributed | failed upstream, nothing attributed it | why |

**`5a` is the only rule other than `7` that can pass a cluster, and it passes it
for a different reason.** It is not a filter but an *adjudication*: by the time
it runs, the cluster has already failed and a specific neighbour has been
blamed. `5a` asks whether the blame holds — separated, and the cluster is
**retrieved**; not separated, and the failure is confirmed. So a `5a` PASS is a
weaker claim than a `7` PASS:

| | means | on 147 real sessions |
|---|---|---|
| `7` `confirmed_pass` | no test fired | 20,397 |
| `5a` `retrieved_neighbor_confirmed` | a test fired and rule 5 overturned it | 3,983 |

**5b and 5c are the only tests in the table that are not feature space**, and
that is deliberate. Scored against per-pair ground truth, the feature-space
statistics are at chance for "one neuron or two" — dip AUC 0.496, L-ratio 0.503,
and a cross-validated linear discriminant 0.503, which is the ceiling for that
family. The cause is circularity: they ask whether two clouds separate in PC
space, and the sorter guaranteed that when it drew the boundary. Spike timing is
not party to that decision, so it can corroborate independently.

**`4b` and `6` are defensive branches and do not fire in practice** — zero of
61,353 clusters across 147 real sessions and 41 synthetic recordings. `4b`
requires an ISI result to be missing, which the pipeline always computes; `6`
handles an LDL verdict of `ambiguous` or `untested`, and `ambiguous` is resolved
to `not_overmerged` before the rules run (the L-ratio takes precedence when the
two statistics conflict) while `untested` has not occurred. They guard records
assembled outside `curate_session`, which is worth keeping and worth stating.

**A caveat that belongs with 5b/5c**: the threshold is a curve in spike count
and it falls steeply — a pair with 200 spikes needs z > 21.9 to be declined, one
with 20,000 needs z > 3.4. On small clusters the time-domain test has little
power, so a `5a` retrieval there rests on the feature-space evidence alone.

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
