# paper/

Code that regenerates every measured panel in the unitassay paper from a
curated corpus. Nothing here is imported by the package; it is a consumer of
`unitassay`'s output, so the library stays free of figure and path code.

Each **data** function takes the corpus root (a directory of
`<session>/curated.json`) and returns a plain dict, so it can be called without
matplotlib and its output diffed between runs. `make_*()` functions only draw;
they never measure.

## Index (figure numbers as in manuscript V9)

| module | figure | what it does | driver script (`claudeSS/code/analysis/`) |
|---|---|---|---|
| `drift_main.py` | **Fig 3** (drift) | main panels: calibration, prevalence, cost, algorithm | `m260928_fig9_main.py` |
| `drift_fig9.py` | **Fig 3** (drift) | single-bank traces and the population (file name predates V9 numbering) | `m260928_fig9.py` |
| `drift_supp.py` | **Supp** (drift) | single-bank corpus and the incidents inside it | `m260925_drift_supp.py`, `m260928_fig9_supp.py` |
| `pipeline_flow.py` | **Fig 5A** | the pipeline as boxes, with each step's measured cost | `m260928_fig5_panelA.py` |
| `layer1_screen.py` | **Fig 7** (layer 1) | channel selection, thresholds, screen vs sorter label, cost | `assemble_figure7_layer1.py` |
| `layer2_rules.py` | **Fig 8** (layer 2) | data: `sieve_flow`, `decision_flow_counts`, exemplar candidates | the four `m260925_fig8_*.py` |
| `layer2_sieve.py` | **Fig 8A** | layer 2 drawn as a sieve | `m260925_fig8_sieve.py` |
| `layer2_flow.py` | **Fig 8A** | layer 2 drawn as a decision flow | `m260925_fig8_sieve.py` |
| `layer2_panels.py` | **Fig 8** rows | one cluster across the six columns of evidence | `m260925_fig8_rank_rows.py`, `_locked_rows.py`, `_contact_sheets.py` |
| `style.py` | all | the manuscript figure style, read from the author's Illustrator file | (imported) |
| `svg_expand.py` | all | turn matplotlib `<use>` markers into standalone paths | (imported) |
| `svg_compose.py` | assembly | stack SVGs into one page without flattening them | (run when assembling a figure) |

Figures 4 (error types) and 6 (benchmark / deployment) are still drawn by
drivers in `claudeSS/code/analysis/` (`m260918_fig3_v2.py`,
`m260928_fig6_v2_*.py`) and have no module here yet.

## Conventions the figures rely on (2026-10-01)

* **Three decisions.** Every cluster's `decision` is `PASS`, `FAIL` or
  `LOW_SPIKE_COUNT` (unitassay `69d20fa`). Count outcomes from `decision`;
  never infer "untestable" from `findings`.
* **Rule labels come from `unitassay.curate.decision_flow`** (`RULE_ID`,
  `RULE_TITLE`, `FIGURE_RULES`), never from a copy here and never from the
  `rule_label` stored in an older run. Rule 6 is not drawn.
* `rule_id` is the rule ENGINE's id, so a LOW_SPIKE_COUNT cluster can still
  carry e.g. `5d`; count by `decision`, label by `rule_id`.
