# paper/

Figure code for the unitassay paper. It turns unitassay's output into the
paper's measured panels, and you can point it at your own recordings to draw
the same panels for them.

Nothing here is imported by the package. `paper/` only consumes what unitassay
writes, so the library itself carries no figure or plotting code.

## 1. Get unitassay

```bash
git clone https://github.com/agritaD/UnitAssay.git
cd UnitAssay
pip install -e .                 # the package: numpy, scipy, scikit-learn, h5py, diptest, click, pyyaml
pip install matplotlib           # needed only for paper/
```

Python 3.10 or newer. Check the install with the bundled four-cluster fixture,
which runs in about a second:

```bash
unitassay run --ks-dir tests/fixture --out /tmp/demo
# 4 clusters   PASS 3  FAIL 1
```

To use the exact code behind the paper's numbers, check out the tag first:

```bash
git checkout v0.1.0-paper
```

## 2. Curate your recordings

`paper/` reads a **corpus**: a directory with one subdirectory per recording,
each holding unitassay's output (`curated.json`, `pairs.json` and
`isolation/`). `unitassay batch` writes exactly that layout:

```bash
# sessions.tsv: one recording per line, name<TAB>path-to-kilosort-output
unitassay batch --sessions sessions.tsv --out my_corpus --n-jobs 6
```

Every `curated.json` records the sorter directory it came from (`ks_dir`). The
drift panels and the per-cluster evidence panels read that directory back, so
keep the sorter output where it was when you curated it.

## 3. Regenerate a panel

Run from the repository root, so that `paper` imports as a package. Each module
has two kinds of function:

- **data functions**, named for the quantity they compute. They take the corpus
  directory (or a sorter directory) and return a plain dict, so you can run them
  without matplotlib and compare their output between runs;
- **`make_*` functions**, which only draw. They take a data function's result and
  an output path, and they never compute a statistic themselves.

```python
from pathlib import Path
from paper import layer2_rules, layer2_flow

corpus = Path('my_corpus')
flow = layer2_rules.decision_flow_counts(corpus)       # data: counts per rule and stage
layer2_flow.make_decision_flow(flow, 'decision_flow.svg')
```

Panels are written as SVG with editable text. `svg_expand.expand_use` converts
repeated markers into ordinary paths so that an illustration program can edit
each point.

## Modules

| module | panels | input |
|---|---|---|
| `layer1_screen.py` | layer 1 (screen): threshold sweep, channel selection, screen against the sorter's label, cost | corpus |
| `layer2_rules.py` | layer 2 data: rule and stage counts, exemplar clusters, correlogram shapes | corpus |
| `layer2_flow.py` | layer 2 drawn as a decision flow | `layer2_rules.decision_flow_counts` |
| `layer2_sieve.py` | layer 2 drawn as a sieve | `layer2_rules.sieve_flow` |
| `layer2_panels.py` | one cluster across the six columns of evidence | corpus + the sorter directory |
| `drift_fig9.py` | single-bank drift traces and drift against recording duration | sorter directories + per-session drift values |
| `drift_main.py` | drift calibration, prevalence by probe geometry, cost, the block algorithm | per-session drift values; a sorter directory |
| `drift_supp.py` | single-bank drift corpus and its artifact incidents | per-session drift values and traces |
| `pipeline_flow.py` | the pipeline as boxes, with each step's measured cost | a list of steps with their times |
| `style.py` | figure style: page width, type sizes, colours | (imported) |
| `svg_expand.py`, `svg_compose.py` | SVG post-processing and assembly | SVG files |

**Per-session drift values** are one dict per recording with `name`, `dur`
(minutes), `clean` (no artifact flag), `cls` (0 single-bank, 1 multi-bank with no
gap block, 2 multi-bank with at least one), `r90` and `r90_pop` (range₉₀ in µm,
with and without gap blocks), and `n_gap_blocks`. The functions in
`unitassay.drift` compute each field from a Kilosort 4 output directory:
`metrics.load_dshift`, `metrics.range_90`, `metrics.artifact_flags` and
`gap_blocks.detect`.

## Conventions

- **Three decisions.** Every cluster's `decision` is `PASS`, `FAIL` or
  `LOW_SPIKE_COUNT`. Count outcomes from `decision`; never infer "untestable" from
  `findings`.
- **Rule labels come from `unitassay.curate.decision_flow`** (`RULE_ID`,
  `RULE_TITLE`, `FIGURE_RULES`), never from a local copy and never from the
  `rule_label` stored by an older run. Rule 6 is not drawn.
- `rule_id` is the rule engine's id, so a `LOW_SPIKE_COUNT` cluster can still
  carry, for example, `5d`. Count by `decision` and label by `rule_id`.
