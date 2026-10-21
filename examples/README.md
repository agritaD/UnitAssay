# Examples

## Quickstart on the example data

The example data are two Kilosort 4 outputs, about 115 MB unzipped:

- `real/`: a 3-minute recording from macaque posterior parietal cortex on a
  Neuropixels 1.0 probe;
- `synthetic/`: a 10-minute MEArec simulation of 102 neurons in close pairs,
  with ground truth for every cluster.

Download `unitassay_example_data.zip` from this repository's Releases page and
unzip it. The raw voltage files are not included; UnitAssay does not need them.

```bash
pip install -e .
python examples/quickstart.py --data path/to/unitassay_example_data
```

It takes a minute or two and walks through what UnitAssay reports:

1. **decisions**: how many clusters are `PASS`, `FAIL` or `LOW_SPIKE_COUNT`
   (too few spikes to test);
2. **why**: the rule that decided each `FAIL`, and the error class it names;
3. **pairs**: duplicate and over-split findings, which involve two clusters;
4. **drift**: how far the probe moved, from Kilosort's own estimate;
5. **ground truth** (synthetic only): sensitivity, specificity and balanced
   accuracy against what each cluster really is.

Each recording is also checked against the reference decisions shipped with the
data (`expected_curated.tsv`). A mismatch means your install behaves differently
from the one behind the paper.

Then browse a result in the viewer:

```bash
unitassay view unitassay_quickstart_out/real
```

## Your own recording

```bash
unitassay run --ks-dir path/to/kilosort4_output --out results
```

The sorter directory needs the standard Kilosort/Phy files. The example's
`kilosort4/` folders list exactly the files UnitAssay reads.
