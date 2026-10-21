# UnitAssay

**Automated post-sorting curation for reliable single-unit identification in high-density electrophysiology.**

UnitAssay is a post-sorting curation framework for evaluating the output of automated spike sorting. It first evaluates cluster isolation and its stability over time, then applies a sequential set of curation tests to identify reliably isolated single units and characterize common spike-sorting errors.

UnitAssay was developed and validated using high-density Neuropixels recordings and Kilosort 4 output. It accompanies the manuscript:

> Agrita Dubey et al., *Scalable hardware and software for high-density, multi-regional electrophysiology in non-human primates.*

A preprint citation and DOI will be added when available.

---

## Installation

UnitAssay requires Python 3.10 or later and runs on CPU; no GPU is required.

From a clone:

```bash
git clone https://github.com/agritaD/UnitAssay.git
cd UnitAssay
pip install .
```

For development:

```bash
pip install -e .
```

UnitAssay does **not** run a spike sorter. Run preprocessing and spike sorting separately, then provide UnitAssay with the sorter output directory.

### Validated pipeline

The pipeline used to develop and benchmark UnitAssay was:

```text
SpikeGLX  ->  CatGT  ->  Kilosort 4  ->  UnitAssay
```

Kilosort 4.0.22 was run with:

```text
Th_universal = 7.5
Th_learned   = 8.5
nblocks      = 5
```

All other Kilosort settings were left at their defaults.

Some UnitAssay thresholds were calibrated using this preprocessing and sorting pipeline. In particular, the overlap-based identity test uses a recording-specific far-pair null that is sensitive to preprocessing. UnitAssay can be run on other compatible Phy-style outputs, but the published performance does not automatically transfer to a different preprocessing or sorting pipeline without validation.

---

## Quick start

Run UnitAssay on one Kilosort output directory:

```bash
unitassay run --ks-dir /path/to/kilosort_output --out /path/to/results
```

Run multiple recordings:

```bash
unitassay batch --sessions sessions.tsv --out /path/to/results --n-jobs 6
```

where `sessions.tsv` contains one recording per line:

```text
session_name<TAB>/path/to/kilosort_output
```

Lines beginning with `#` are treated as comments.

From Python:

```python
from unitassay import curate

rows = curate.curate_session(
    "/path/to/kilosort_output",
    isolation_dir="isolation",
)

clean = [row for row in rows if row["outcome"] == "clean"]
```

---

## Input

UnitAssay reads a standard Phy-style sorter directory containing:

```text
spike_times.npy
spike_clusters.npy
pc_features.npy
pc_feature_ind.npy
templates.npy
channel_positions.npy
params.py
```

The raw voltage file is not required for the main curation workflow.

---

## What UnitAssay does

UnitAssay has two main curation layers.

### 1. Screen

The screening layer evaluates cluster isolation and its stability over time using L-ratio-based comparisons. Each cluster is evaluated against a shape-dissimilar reference channel and against neighboring clusters.

### 2. Curate

The curation layer applies a sequential decision flow that evaluates five principal sources of evidence:

1. **Identity / re-detection** — whether the same spikes have been counted under another cluster.
2. **Refractory structure** — whether the spike train is compatible with a single neuron.
3. **Within-cluster composition** — whether the cluster contains separable sub-populations.
4. **Isolation / contamination** — whether the cluster remains isolated relative to a reference channel.
5. **Neighbor relationships** — whether an apparent isolation failure is supported or overturned by comparisons with neighboring clusters.

Additional guard and stability branches handle incomplete evidence and failures of isolation stability.

The decision flow separates the **test that fired** from the **error class assigned to the cluster**. This distinction is deliberate: a measured feature such as bimodality or contamination does not, by itself, establish the biological origin of the error.

---

## Output

Each run writes a cluster-level curation output (`curated.json`) and pair-level output (`pairs.json`).

### Cluster-level decisions

The main cluster-level decision is:

- `PASS` — accepted as a usable single unit by the curation flow.
- `FAIL` — rejected by the curation flow.
- `LOW_SPIKE_COUNT` — below the spike-count floor required for the curation tests.

The output also records the rule that issued the decision, the evidence produced by individual tests, tests that could not run, and a higher-level outcome label.

### Outcome labels

The current reporting taxonomy includes:

| outcome | interpretation |
|---|---|
| `clean` | no curation criterion identified a problem |
| `duplicate` | spikes are already represented by another cluster |
| `oversplit` | cluster is most consistent with a fragment of another unit |
| `mixed_cluster` | more than one population is present, but the underlying cause is not resolved |
| `intermittent_presence` | isolation is not maintained across the recording |
| `low_spike_count` | too few spikes for the full curation procedure |

`mixed_cluster` is intentionally conservative. UnitAssay does not label these clusters as over-merged when the available evidence cannot reliably distinguish an over-merge from other causes of within-cluster heterogeneity.

### Pair-level output

Duplication and over-splitting are inherently pairwise. UnitAssay therefore writes `pairs.json` in addition to the per-cluster output.

Pair-level records preserve the two cluster IDs, the comparison channel, whether the criterion was met, the measured evidence, and the recommended action. This allows pairwise evidence to be inspected without forcing it into a single-cluster label.

---

## Example data

A small synthetic fixture ships with the repository and can be used as a smoke test:

```bash
unitassay run --ks-dir tests/fixture --out /tmp/unitassay_demo
```

The fixture is intended to verify that the installation and decision flow run correctly. It is **not** an accuracy benchmark.

Two larger example Kilosort 4 outputs — one real recording and one synthetic recording with ground truth — are provided through the repository Releases page. See [examples/README.md](examples/README.md) for the quick-start workflow.

```bash
python examples/quickstart.py --data /path/to/unitassay_example_data
```

The example workflow reports curation decisions, the evidence behind failures, pairwise findings, drift estimates, and ground-truth performance for the synthetic example.

---

## Performance

UnitAssay was benchmarked on 28 synthetic MEArec recordings sorted with Kilosort 4. The benchmark contained 3,664 clusters with at least 240 spikes, including 1,134 ground-truth clean clusters.

All methods below were evaluated on the same set of clusters.

| method | sensitivity | specificity | balanced accuracy |
|---|---:|---:|---:|
| Kilosort 4 `good` label | 82.5% | 58.2% | 70.4% |
| SpikeInterface QC | 82.5% | 58.7% | 70.6% |
| **UnitAssay PASS** | **78.6%** | **85.8%** | **82.2%** |

SpikeInterface QC used ISI ratio ≤ 0.5, presence ratio ≥ 0.8, and amplitude cutoff ≤ 0.1.

These values describe performance on the synthetic benchmark and validated preprocessing/sorting pipeline above. They should not be assumed to transfer unchanged to other probes, preparations, sorters, or preprocessing pipelines.

### Computational cost

Across 147 real recording sessions containing 53,346 clusters, the median processing time was 92 ms per cluster for screening and 74 ms per cluster for curation. The combined workflow required a median of 63 s per session in the benchmarked configuration.

---

## Additional drift utilities

The repository also includes utilities for working with Kilosort drift estimates on multi-bank probe configurations.

```python
from unitassay.drift import gap_blocks

report = gap_blocks.exclude_gap_blocks("/path/to/kilosort_output")
print(report["range_90_gap_excluded"])
```

These utilities identify Kilosort drift blocks that fall in sparsely populated depth gaps in the channel map and report drift summaries with those blocks excluded.

**Exclusion is not correction.** If a block lies in a depth range without recording sites, UnitAssay can flag or exclude that estimate, but it cannot reconstruct the missing drift measurement. Correcting the underlying behavior requires changes to the sorter and re-sorting.

---

## Limitations

- **Validation is pipeline-specific.** UnitAssay was calibrated and benchmarked using CatGT preprocessing and Kilosort 4. Other pipelines require independent validation.
- **Ground truth is synthetic.** Accuracy metrics come from synthetic recordings. Real recordings do not provide cluster-level ground truth.
- **Mixed clusters are not assigned a specific cause when the evidence is insufficient.** In particular, UnitAssay does not claim that within-cluster heterogeneity necessarily reflects an over-merge.
- **Low-spike-count clusters are not fully adjudicated.** The current floor is 240 spikes because the within-cluster comparison requires two groups with sufficient samples.
- **Drift is identified and annotated, not corrected.**
- **Pairwise errors require pairwise interpretation.** Per-cluster labels cannot fully represent all duplicate or over-split relationships.

---

## Reproducibility

The paper analyses use the UnitAssay version tagged:

```text
v0.1.0-paper
```

The figure code, synthetic-data recipe, and ground-truth analysis used for the manuscript are maintained separately in the companion research repository `MIND_UnitAssay_paper`.

For published analyses, use the paper tag rather than the moving `main` branch.

---

## Authors

Developed by **Agrita Dubey** and **Qiao (Betty) Xu**  
Pesaran Lab, University of Pennsylvania

---

## License

UnitAssay is released under the MIT License. See [LICENSE](LICENSE).
