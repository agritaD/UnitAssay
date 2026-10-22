#!/usr/bin/env python3
"""curate/lratio.py

Canonical SINGLE-DIRECTION L-ratio: how much does `pc_other`'s spike cloud
intrude into a Gaussian fit to `pc_unit`'s cloud. This is the screen layer's
contamination question -- "if I call these spikes one unit, how much does
some other spike source contaminate it" -- and mirrors
unitassay/screen/curation/auto_curate.py._compute_l_ratio exactly (unit is always the
reference; not symmetric).

Consolidated 2026-07-13 from 13 near-identical copies (_ms_l_ratio,
_l_ratio, _compute_l_ratio, l_ratio_frame) across noise-fail /
neighbor-fail analysis scripts.

NOT the same formula as curate/bimodality.py's mutual L-ratio (max over
BOTH directions, used to ask "are these two clusters mutually separable"
for overmerge/oversplit testing) -- that's a genuinely different question
(symmetric separability vs. asymmetric contamination-of-a-reference) and
deliberately stays a separate function; do not merge the two.

Author: Agrita Dubey
Part of unitassay. Licence: MIT.
"""

import numpy as np
from scipy.stats import chi2 as _chi2


def ms_l_ratio(pc_unit, pc_other):
    """L-ratio of pc_other against a Gaussian fit to pc_unit (screen layer /
    Schmitzer-Torbert & Redish 2004 formula).

    pc_unit: (n_unit, p) flat feature matrix, the REFERENCE cluster.
    pc_other: (n_other, p) flat feature matrix, the cluster being tested
              for contamination of pc_unit's region. Empty/None -> 0.0
              (no competing spikes = fully isolated).

    Returns np.nan if pc_unit has too few spikes / degenerate covariance.
    """
    if pc_unit is None or pc_unit.ndim < 2 or pc_unit.shape[0] <= pc_unit.shape[1] or pc_unit.shape[0] < 4:
        return np.nan
    if pc_other is None or pc_other.ndim < 2 or pc_other.shape[0] == 0:
        return 0.0
    try:
        n, p = pc_unit.shape
        mu = np.mean(pc_unit, axis=0)
        cov = np.cov(pc_unit, rowvar=False, ddof=1) + np.eye(p) * 1e-9
        inv = np.linalg.inv(cov)
        d2 = np.einsum('ij,jk,ik->i', pc_other - mu, inv, pc_other - mu)
        return float(np.sum(1 - _chi2.cdf(d2, p)) / n)
    except Exception:
        return np.nan
