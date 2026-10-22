"""screen/curation/overmerge.py — the adopted overmerge test (LDL).

Port of `unitassay.curate.bimodality.overmerge_verdict`, kept SELF-CONTAINED
so the screen layer does not import from the curate layer. The two copies must
agree; if the math changes in one, change it in the other.

WHY THIS REPLACED THE PREVIOUS `_is_lda_bimodal`. That was a port of the RETIRED
forced-k-means(2) design, and it stayed in the shipped package long after the
analysis side moved on -- the sixth of six near-duplicate implementations, and
the only one still running in production. Measured consequences of the old one
on the screen layer's burst gate:

  * it reads a LOPSIDED split as two units, so its error is a false POSITIVE
  * on 2,417 isi_vr clusters it wrongly BLOCKED 11 burst rescues and wrongly
    allowed ZERO -- the opposite direction from what was assumed for months
  * the harm was masked, because curate() checks overmerge before ISI-VR, so a
    wrongly-blocked rescue usually failed for another reason anyway

THE DESIGN, in one paragraph. Impose ONE 2-component diagonal GMM (never a
best-k sweep). If the split collapses, or the minority is a negligible satellite,
or the minority cannot support an L-ratio, stop -- those are results, not
absences. Otherwise recurse while the DIP alone still finds structure, which
yields "leaves" (groups nothing can split further); then score every leaf pair
that can support an L-ratio and take the most separated one. A cluster is
overmerged when that pair has dip p < 0.05 AND L-ratio < 0.05.

The dip drives the recursion; the L-ratio decides. Both matter: on ground truth
the L-ratio is right 256/256 when the two disagree on known single units, but
the dip is what finds the structure in the first place.

Authors: Agrita Dubey, Qiao (Betty) Xu
Part of unitassay. Licence: MIT.
"""
from __future__ import annotations

import numpy as np
import scipy.spatial.distance as _spdist
from scipy.stats import chi2 as _chi2
from sklearn.mixture import GaussianMixture

try:
    import diptest
except ImportError as e:  # pragma: no cover
    raise ImportError("The overmerge test requires the 'diptest' package") from e

DIP_PVAL_THRESH = 0.05
L_RATIO_THRESH = 0.05
MIN_SPIKES = 50
# An L-RATIO requirement, not a dip requirement: it inverts each group's OWN
# covariance, degenerate at n~40 in d=30. Gating the RECURSION on it is what
# once stopped the search on lopsided splits and left the large side unexamined.
MIN_GROUP_SPIKES = 120
SATELLITE_RATE_HZ = 0.1
MAX_DEPTH = 3


def _lda_direction(fa, fb, d):
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    Sw = ((len(fa) - 1) * np.cov(fa.T) + (len(fb) - 1) * np.cov(fb.T)) / (
        len(fa) + len(fb) - 2)
    reg = max(1e-6, 1e-4 * np.trace(Sw) / d) * np.eye(d)
    try:
        w = np.linalg.solve(Sw + reg, mu_b - mu_a)
    except np.linalg.LinAlgError:
        return None, reg
    nrm = np.linalg.norm(w)
    return (None, reg) if nrm == 0 else (w / nrm, reg)


def _mutual_l_ratio(fa, fb, d, reg):
    """Both directions, the more conservative winning."""
    vals = []
    for pcs_self, pcs_other in ((fa, fb), (fb, fa)):
        try:
            VI = np.linalg.inv(np.cov(pcs_self.T) + reg)
            mu_s = pcs_self.mean(0, keepdims=True)
            mah = _spdist.cdist(mu_s, pcs_other, "mahalanobis", VI=VI)[0]
            vals.append(float(np.sum(1 - _chi2.cdf(mah ** 2, df=d)) / len(pcs_self)))
        except Exception:
            return np.nan
    return float(max(vals))


def dip_lratio_between(fa, fb, min_spikes=MIN_SPIKES, min_group_spikes=None):
    fail = {"dip_pval": np.nan, "l_ratio": np.nan, "proj": None, "w": None}
    floor = max(min_spikes // 4, 2)
    if min_group_spikes is not None:
        floor = max(floor, min_group_spikes)
    if fa.shape[0] < floor or fb.shape[0] < floor:
        return fail
    d = fa.shape[1]
    if d < 2 or fb.shape[1] != d:
        return fail
    w, reg = _lda_direction(fa, fb, d)
    if w is None:
        return fail
    proj = np.concatenate([fa @ w, fb @ w])
    _, dip_pval = diptest.diptest(proj)
    return {"dip_pval": float(dip_pval), "l_ratio": _mutual_l_ratio(fa, fb, d, reg),
            "proj": proj, "w": w}


def _atomize(X, depth, min_spikes, min_group_spikes, dip_thresh, max_depth,
             info=None):
    """Split while the DIP still finds structure. Returns the leaves."""
    if depth >= max_depth or X.shape[0] < 8:
        return [X]
    try:
        lab = GaussianMixture(n_components=2, covariance_type="diag",
                              random_state=0, n_init=1).fit(X).predict(X)
    except Exception:
        return [X]
    fa, fb = X[lab == 0], X[lab == 1]
    if min(len(fa), len(fb)) < 4:
        return [X]
    # min_group_spikes=None on purpose: the dip is a 1-D test and is valid far
    # below the L-ratio's floor, so it can gate a split the L-ratio could not
    # adjudicate. The floor is enforced later, where it belongs.
    r = dip_lratio_between(fa, fb, min_spikes=min_spikes, min_group_spikes=None)
    if r["w"] is None or np.isnan(r["dip_pval"]) or r["dip_pval"] >= dip_thresh:
        # ATOMIC, and the dip that established it is EVIDENCE. Discarding it is
        # why clusters used to report "no dip result" despite the dip having run
        # and answered "no internal structure".
        if depth == 0 and info is not None:
            info["root_dip"] = (np.nan if r["w"] is None
                                or not np.isfinite(r["dip_pval"])
                                else float(r["dip_pval"]))
        return [X]
    return (_atomize(fa, depth + 1, min_spikes, min_group_spikes, dip_thresh,
                     max_depth)
            + _atomize(fb, depth + 1, min_spikes, min_group_spikes, dip_thresh,
                       max_depth))


def _best_leaf_pair(leaves, min_spikes, min_group_spikes):
    best = None
    floor = max(min_group_spikes or 0, 4)
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            fa, fb = leaves[i], leaves[j]
            if min(len(fa), len(fb)) < floor:
                continue
            r = dip_lratio_between(fa, fb, min_spikes=min_spikes,
                                   min_group_spikes=min_group_spikes)
            if r["w"] is None or np.isnan(r["dip_pval"]) or np.isnan(r["l_ratio"]):
                continue
            key = (float(r["l_ratio"]), float(r["dip_pval"]))
            if best is None or key < best[0]:
                best = (key, float(r["dip_pval"]), float(r["l_ratio"]))
    return best


def overmerge_verdict(feats, min_spikes=MIN_SPIKES, min_group_spikes=MIN_GROUP_SPIKES,
                      duration_sec=None, satellite_rate_hz=SATELLITE_RATE_HZ,
                      dip_thresh=DIP_PVAL_THRESH, l_thresh=L_RATIO_THRESH,
                      max_depth=MAX_DEPTH):
    """Does this cluster hold more than one unit?

    Returns dict: verdict ('overmerged' | 'not_overmerged' | 'ambiguous' |
    'insufficient'), basis, dip_pval, l_ratio, n_leaves, parked, resolved.
    `resolved` is False when no comparison could be formed -- absence of a
    verdict, which must never be read as a pass.
    """
    n, d = feats.shape
    base = {"verdict": "insufficient", "basis": "none", "dip_pval": np.nan,
            "l_ratio": None, "n_leaves": 0, "parked": None, "resolved": False}
    if n < min_spikes or d < 2:
        return base
    try:
        lab0 = GaussianMixture(n_components=2, covariance_type="diag",
                               random_state=0, n_init=1).fit(feats).predict(feats)
    except Exception:
        return base
    g0, g1 = feats[lab0 == 0], feats[lab0 == 1]
    n_small = min(len(g0), len(g1))

    work, parked, basis = feats, None, "full"
    if n_small < 4:
        basis = "degenerate_split"
    elif duration_sec and (n_small / duration_sec) < satellite_rate_hz:
        basis = "negligible_satellite"
    elif n_small < min_group_spikes:
        basis = "lopsided_no_lratio"

    # Statistics of the FIRST split, kept so a `full+...` outcome reports the
    # evidence it was reached on. The analysis-side copy carries these forward
    # via {**first}; omitting them here made the two disagree on every atomic
    # case while agreeing on every verdict -- caught by the parity check.
    first_dp, first_lr = np.nan, None
    if basis == "full":
        r0 = dip_lratio_between(g0, g1, min_spikes=min_spikes,
                                min_group_spikes=min_group_spikes)
        if r0["w"] is not None and not np.isnan(r0["dip_pval"]) \
                and not np.isnan(r0["l_ratio"]):
            first_dp, first_lr = float(r0["dip_pval"]), float(r0["l_ratio"])

    if basis != "full":
        big, small = (g0, g1) if len(g0) >= len(g1) else (g1, g0)
        if big.shape[0] < max(min_group_spikes, 4) * 2:
            return {**base, "verdict": "not_overmerged",
                    "basis": basis + "+remainder_too_small",
                    "n_leaves": 1, "parked": int(small.shape[0]), "resolved": False}
        work, parked = big, int(small.shape[0])

    atom_info = {}
    leaves = _atomize(work, 0, min_spikes, min_group_spikes, dip_thresh, max_depth,
                      info=atom_info)
    best = _best_leaf_pair(leaves, min_spikes, min_group_spikes)
    if best is None:
        atomic = len(leaves) <= 1
        tail = ("+atomic" if atomic else "+unadjudicable") if parked is None else \
               ("+large_atomic" if atomic else "+large_unadjudicable")
        # Report the dip that established atomicity when the first split exited
        # early and so never recorded one.
        dp_atomic = first_dp
        if (dp_atomic is None or not np.isfinite(dp_atomic)) and atomic:
            dp_atomic = atom_info.get("root_dip", dp_atomic)
        return {**base, "verdict": "not_overmerged", "basis": basis + tail,
                "dip_pval": dp_atomic, "l_ratio": first_lr,
                "n_leaves": len(leaves), "parked": parked, "resolved": bool(atomic)}
    _, dp, lr = best
    if dp < dip_thresh and lr < l_thresh:
        v = "overmerged"
    elif dp >= dip_thresh and lr >= l_thresh:
        v = "not_overmerged"
    else:
        v = "ambiguous"
    return {**base, "verdict": v,
            "basis": ("recursive" if parked is None else basis + "+large_tested"),
            "dip_pval": dp, "l_ratio": lr, "n_leaves": len(leaves),
            "parked": parked, "resolved": True}


def is_overmerged(feats, min_spikes=MIN_SPIKES, duration_sec=None) -> bool:
    """The burst gate's question: should the ISI-VR rescue be BLOCKED?

    True only on a confident `overmerged` verdict. Every other outcome --
    ambiguous, unresolved, insufficient, any failure -- returns False, keeping
    the original conservative default of not blocking a rescue on shaky
    evidence.

    PASS `duration_sec`. Without it the `negligible_satellite` branch cannot run
    at all, because it is a RATE test (n_small / duration < 0.1 Hz) and a None
    duration makes that condition unconditionally false. Two consequences, and
    the second is a real error:

      * a genuine sub-0.1 Hz satellite is relabelled `lopsided_no_lratio`, which
        parks it identically -- harmless, but it hides the mechanism from every
        tally. Measured on the 50-session set: negligible_satellite was 0.0% of
        18,974 clusters while lopsided_no_lratio was 37.4%.
      * when the minority holds >= min_group_spikes AND fires below the rate, it
        is TESTED as a candidate second unit instead of parked, which can
        manufacture a false `overmerged` verdict and so wrongly BLOCK a burst
        rescue. That band is [min_group_spikes, 0.1*T), empty unless T > 1200 s.

    Left defaulting to None rather than made required so existing callers do not
    break, but every caller that can supply a duration must.
    """
    try:
        return overmerge_verdict(feats, min_spikes=min_spikes,
                                 duration_sec=duration_sec)["verdict"] == "overmerged"
    except Exception:
        return False
