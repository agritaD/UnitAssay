"""Auto-curation using L-Ratio and ISI violation thresholds.

Ports auto_curate.m. Classifies units as isolated or non-isolated
based on Mahalanobis-based L-Ratio and ISI violation rate.

EXPERIMENTAL (this dev copy only, 2026-07-14): ISI violation rate now has a
burst-cell correction ported from analysis/compute_corrected_isi.py's
is_burst_cell() (same logic, same 0.80 threshold, same 1ms refractory
period -- not a window change, just a guard on top of it). A cluster that
fails ISI-VR but has >=80% of its <1ms violations concentrated below 0.5ms
looks like an intrinsic burst-firing neuron (doublets/triplets), not
contamination from a second neuron, so the ISI-VR failure is overridden and
isolation falls back to L-Ratio alone -- see todo_mindsnag_burst_fix.md
Option A.

GATED (added same day, 2026-07-14, after re-reading
m260708_overmerging_three_cases.py / Fig_Overmerging_ThreeCases.png): the
short-ISI-concentration signature alone CANNOT tell a burst cell apart from
a genuine overmerged cluster -- that figure's row 1 (confirmed genuine
overmerge, dip_p=0.0005) has 95.2% of its <1ms violations below 0.5ms,
comfortably over the 80% threshold, indistinguishable from row 2's true
burst cell (dip_p=0.9507) by timing alone. Only the multichannel LDA
dip-test + L-ratio bimodality check -- standard name (2026-07-14): the "LDL
test" (LDA-Dip-Lratio; Utils/bimodality.py's lda_dip_lratio+verdict in the
claudeSS analysis scripts; reimplemented self-contained below as
_is_lda_bimodal since this package doesn't import from claudeSS/code/Utils/
-- keep in sync with that canonical file) tells them apart. So the
burst-cell override now additionally requires the cluster's own multichannel
PC features (NPclu.h5's stored pcFeat tensor) to come back UNIMODAL --
dip_pval >= 0.05 OR l_ratio >= 0.05 -- before the ISI-VR failure is forgiven.
A cluster that is both burst-concentrated AND LDA-bimodal keeps failing
ISI-VR, matching that figure's conclusion exactly.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from unitassay.isolate.curation.overmerge import is_overmerged
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import chi2
from sklearn.cluster import KMeans

try:
    import diptest
except ImportError as e:
    raise ImportError("mind_snag's burst-cell LDA gate requires the 'diptest' package") from e

from unitassay.isolate.config import IsolateConfig

logger = logging.getLogger(__name__)

REFRACTORY_PERIOD = 0.001  # 1 ms
BURST_FRAC_THRESH = 0.80   # >=80% of <1ms violations at <0.5ms -> burst cell, not contamination
LDA_MIN_SPIKES = 50
LDA_N_PC = 3
LDA_DIP_PVAL_THRESH = 0.05
LDA_L_RATIO_THRESH = 0.05


def _is_lda_bimodal(feats: np.ndarray, min_spikes: int = LDA_MIN_SPIKES,
                    duration_sec: float | None = None) -> bool:
    """Should the ISI-VR burst rescue be BLOCKED because this is an overmerge?

    REPLACED 2026-08-13. This used to be a self-contained port of the RETIRED
    forced-k-means(2) design, and it stayed here long after the analysis side
    moved on -- the last of six near-duplicate implementations, and the only one
    still running in production. Measured on 2,417 isi_vr clusters, its error was
    a false POSITIVE: it read a LOPSIDED split as two units and so wrongly
    BLOCKED 11 burst rescues while wrongly allowing zero.

    Now delegates to mind_snag.curation.overmerge, which implements the adopted
    recursive dip + L-ratio test. Kept as a thin wrapper rather than deleted so
    the call sites below are untouched. Name retained for the same reason,
    though it now means "is overmerged" rather than "is LDA-bimodal".

    Behaviour on failure is unchanged: anything other than a confident
    `overmerged` verdict returns False, so a rescue is never blocked on shaky
    evidence.

    `duration_sec` added 2026-08-20 and must be supplied: the satellite filter is
    a RATE test, so a None duration silently disables it -- see is_overmerged().
    """
    return is_overmerged(feats, min_spikes=min_spikes, duration_sec=duration_sec)

def auto_curate(output_dir: str | Path, cfg: IsolateConfig) -> None:
    """Auto-curate clusters based on L-Ratio and ISI violation thresholds.

    For each cluster's SortData:
    1. Computes L-Ratio (Mahalanobis) between unit PCs and noise + neighbor
       cluster PCs, PER TIME FRAME. Passes if the fraction of frames where
       all L-Ratios pass exceeds cfg.curation.isolated_t_ratio. NaN L-Ratio
       (no data in a frame) counts as passing that frame -- no evidence of
       contamination means it isn't penalized.
    2. Checks ISI violation rate < cfg.curation.isi_violation_rate

    Sets UnitIso = 1 for passing clusters.

    Reverted to per-frame (2026-07-16) after a pooled-L-ratio experiment
    (2026-07-15) was tested at population scale and found net-negative
    against the best-k LDL test (132 regressions vs 77 improvements among
    the 247 clusters that flip verdict) -- see [[spikesorting-ldl-test-audit]]
    memory for the full comparison. Pooling did fix the temporally-
    segregated missed-overmerge case it targeted, but broke more previously-
    correct passes than it fixed, for a reason not fully identified (not
    simply large-N, per the same memory).

    Parameters
    ----------
    output_dir : directory containing NPclu.h5 and SortData files
    cfg : pipeline configuration with curation thresholds
    """
    output_dir = Path(output_dir)

    l_thresh = cfg.curation.l_ratio_threshold
    isi_thresh = cfg.curation.isi_violation_rate
    t_ratio = cfg.curation.isolated_t_ratio

    logger.info(
        "Auto-curation: L-Ratio=%.2f, ISI=%.2f, t-Ratio=%.2f",
        l_thresh, isi_thresh, t_ratio,
    )

    # Load NPclu for spike times and cluster IDs
    npclu_file = output_dir / "NPclu.h5"
    if not npclu_file.exists():
        logger.warning("NPclu.h5 not found in %s", output_dir)
        return

    with h5py.File(npclu_file, "r") as f:
        # Stored [2 x N], transpose to [N x 2] for MATLAB-style indexing
        npclu = np.array(f["NPclu"]).T
        spiketimes_all = npclu[:, 0]
        spike_templates_all = npclu[:, 1].astype(int)
        clu_info = np.array(f["Clu_info"]).T  # [nClu x 2]
        # Multichannel PC features for the burst-vs-overmerge LDA gate, only
        # needed lazily per-cluster below -- [nLocal x 3 x N] on disk (MATLAB
        # layout), transposed to [N x 3 x nLocal] for Python indexing.
        pc_feat_all = np.array(f["pcFeat"]).transpose(2, 1, 0) if "pcFeat" in f else None

    sub_clus = clu_info[:, 0].astype(int)

    # Find all SortData files
    sort_files_h5 = {int(f.stem.split(".")[0]): f for f in output_dir.glob("*.SortData.h5")}

    # Reset all unit_iso to 0 before re-curation so stale results don't persist
    for sort_file in sort_files_h5.values():
        try:
            with h5py.File(sort_file, "r+") as f:
                if "frame_0000" in f:
                    f["frame_0000"].attrs["unit_iso"] = 0
        except OSError:
            pass

    n_isolated = 0
    n_processed = 0
    n_burst_corrected = 0

    for clu_id in sub_clus:
        sort_file = sort_files_h5.get(clu_id)
        if sort_file is None:
            logger.debug("SortData not found for cluster %d", clu_id)
            continue

        with h5py.File(sort_file, "r+") as f:
            n_frames = f.attrs.get("n_frames", 0)
            if n_frames == 0:
                continue

            # Get neighbor cluster IDs
            other_ids = []
            if "frame_0000" in f:
                frame0 = f["frame_0000"]
                if "other_clu" in frame0:
                    other_ids = np.array(frame0["other_clu"]).astype(int).tolist()
                elif "other_clu" in frame0.attrs:
                    other_ids = np.atleast_1d(frame0.attrs["other_clu"]).astype(int).tolist()

            # Pre-load neighbor SortData
            other_sort_data = {}
            for oid in other_ids:
                ofile = sort_files_h5.get(oid)
                if ofile is not None:
                    other_sort_data[oid] = ofile

            # --- Criterion 1: L-Ratio + temporal consistency, PER FRAME ---
            # Restored 2026-07-16 (reverted the 2026-07-15 pooled-L-ratio
            # experiment -- see the module docstring above and
            # [[spikesorting-ldl-test-audit]] memory for why). L-ratio
            # computed independently per time frame for both the noise-
            # channel comparison and any same-peak-channel neighbor; passes
            # if the fraction of frames where ALL comparisons pass meets
            # cfg.curation.isolated_t_ratio.
            n_other = len(other_ids)
            l_ratio_matrix = np.full((1 + n_other, n_frames), np.nan, dtype=np.float32)

            for i_frame in range(n_frames):
                frame_key = f"frame_{i_frame:04d}"
                if frame_key not in f:
                    continue
                frame = f[frame_key]

                if "unit" not in frame:
                    continue
                pc_unit = np.array(frame["unit"])
                if pc_unit.ndim < 2 or pc_unit.shape[0] <= pc_unit.shape[1]:
                    continue  # too few spikes

                if "noise" in frame:
                    pc_noise = np.array(frame["noise"])
                    l_ratio_matrix[0, i_frame] = _compute_l_ratio(pc_unit, pc_noise)

                for i_o, oid in enumerate(other_ids):
                    ofile_path = other_sort_data.get(oid)
                    if ofile_path is None:
                        continue
                    with h5py.File(ofile_path, "r") as of:
                        oframe_key = f"frame_{i_frame:04d}"
                        if oframe_key not in of or "unit" not in of[oframe_key]:
                            continue
                        pc_other = np.array(of[oframe_key]["unit"])
                        l_ratio_matrix[1 + i_o, i_frame] = _compute_l_ratio(pc_unit, pc_other)

            # For frames where unit had data, missing neighbor data = isolated (0)
            unit_has_data = ~np.isnan(l_ratio_matrix[0, :])
            for i_o in range(n_other):
                nan_mask = np.isnan(l_ratio_matrix[1 + i_o, :]) & unit_has_data
                l_ratio_matrix[1 + i_o, nan_mask] = 0.0

            # NaN in row 0 = unit had no spikes in that frame = drifted away = not isolated.
            # NaN in rows 1+ already converted to 0 above (neighbor absent = no contamination).
            iso_filter = np.all(l_ratio_matrix < l_thresh, axis=0)
            iso_frame_ratio = float(np.sum(iso_filter)) / n_frames
            passes_lratio = iso_frame_ratio >= t_ratio

            # --- Criterion 2: ISI violation rate (burst-cell corrected) ---
            passes_isi = True
            viol_rate = np.nan
            is_burst = False
            my_spikes = np.where(spike_templates_all == clu_id)[0]
            if len(my_spikes) > 0:
                st = np.sort(spiketimes_all[my_spikes])
                isis = np.diff(st)
                N = len(st)
                T = st[-1] - st[0]
                if T > 0 and N > 1:
                    lam = N / T
                    num_viol = int(np.sum(isis < REFRACTORY_PERIOD))
                    if num_viol < 5:
                        # Fewer than 5 absolute violations: insufficient evidence
                        # to flag (avoids false positives on low firing rate
                        # neurons where the Hill denominator, expected_viol, is
                        # tiny -- a handful of raw violations can otherwise blow
                        # the normalized rate past threshold). Present in copy2
                        # (Drift_Figure_Code/mind_snag) but dropped here when
                        # burst-correction was added, apparently unintentionally
                        # -- the burst-correction gate below does NOT protect
                        # this case (it targets timing-concentrated violations
                        # from a genuinely bursting high-rate neuron, a
                        # different failure mode). Restored 2026-07-26 after
                        # Agrita traced a 50-session regression to this exact
                        # gap: 582/582 affected clusters had num_viol<5 and
                        # ZERO were caught by the burst-correction rescue.
                        viol_rate = 0.0
                    else:
                        expected_viol = 2 * REFRACTORY_PERIOD * lam * N
                        viol_rate = num_viol / expected_viol if expected_viol > 0 else 0.0
                    passes_isi = viol_rate < isi_thresh

                    if not passes_isi:
                        # Burst-cell correction: same 1ms REFRACTORY_PERIOD,
                        # just an override on top -- if the violations are
                        # concentrated in the <0.5ms sub-window, this LOOKS
                        # like an intrinsic burst-firing neuron. But timing
                        # alone can't tell that apart from a genuine
                        # overmerged cluster (Fig_Overmerging_ThreeCases.png
                        # row 1: 95.2% concentration, confirmed genuine
                        # overmerge) -- so also require the multichannel LDA
                        # dip+L-ratio test to come back NOT bimodal before
                        # forgiving the ISI-VR failure. If it IS bimodal,
                        # this is a real overmerge and must keep failing.
                        isis_ms = isis * 1000.0
                        viol_1ms = isis_ms[isis_ms < 1.0]
                        burst_like = len(viol_1ms) > 0 and np.mean(viol_1ms < 0.5) >= BURST_FRAC_THRESH
                        if burst_like:
                            bimodal = False
                            if pc_feat_all is not None:
                                n_pc_avail = min(LDA_N_PC, pc_feat_all.shape[1])
                                feats = pc_feat_all[my_spikes, :n_pc_avail, :].reshape(len(my_spikes), -1)
                                # spiketimes_all is in SECONDS (isis below are
                                # scaled by 1000 to reach ms), so this is the
                                # recording duration the satellite RATE test
                                # needs. Omitting it disabled that test.
                                bimodal = _is_lda_bimodal(
                                    feats, duration_sec=float(spiketimes_all.max()))
                            if not bimodal:
                                is_burst = True
                                passes_isi = True

            # --- Final decision ---
            is_isolated = passes_lratio and passes_isi
            f["frame_0000"].attrs["unit_iso"] = int(is_isolated)

        n_processed += 1
        n_isolated += int(is_isolated)
        n_burst_corrected += int(is_burst)

        logger.info(
            "  Clu %3d: Iso=%d  LR_pass=%.2f%%  ISI_VR=%.6f%s",
            clu_id, int(is_isolated), iso_frame_ratio * 100,
            viol_rate if not np.isnan(viol_rate) else 0.0,
            "  [burst-corrected]" if is_burst else "",
        )

    logger.info("%d/%d clusters isolated (%d burst-corrected ISI-VR overrides)",
                n_isolated, n_processed, n_burst_corrected)


def _compute_l_ratio(pc_unit: np.ndarray, pc_other: np.ndarray) -> float:
    """Mahalanobis-based L-Ratio between unit and other (noise/cluster) spikes.

    Matches MATLAB compute_l_ratio_local exactly:
        all_spikes = [unit; other]
        D2 = mahal(all_spikes, unit)       % MATLAB mahal
        other_D2 = D2(n_unit+1:end)
        L = sum(1 - chi2cdf(other_D2, p)) / n_unit

    MATLAB's mahal(Y, X) computes: (Y - mean(X)) * inv(cov(X)) * (Y - mean(X))'
    using the covariance of X (the reference group), applied to each row of Y.
    """
    if pc_unit is None or len(pc_unit) == 0 or pc_unit.shape[0] <= pc_unit.shape[1]:
        return np.nan  # can't assess unit
    if pc_other is None or len(pc_other) == 0:
        return 0.0  # no competing spikes = isolated

    try:
        n_unit = pc_unit.shape[0]
        p = pc_unit.shape[1]
        all_spikes = np.vstack([pc_unit, pc_other])

        # Match MATLAB mahal(): covariance of the reference (unit) group
        mean_unit = np.mean(pc_unit, axis=0)
        # MATLAB cov() uses N-1 normalization
        cov_unit = np.cov(pc_unit, rowvar=False, ddof=1)

        # Regularize if near-singular (matches MATLAB behavior with eps)
        cov_unit += np.eye(p) * np.finfo(float).eps

        cov_inv = np.linalg.inv(cov_unit)

        # Mahalanobis distance squared for ALL points against unit distribution
        diff = all_spikes - mean_unit
        d2 = np.sum((diff @ cov_inv) * diff, axis=1)

        # Take only the other (non-unit) distances
        other_d2 = d2[n_unit:]

        l_sum = np.sum(1 - chi2.cdf(other_d2, p)) + np.finfo(float).eps
        return float(l_sum / n_unit)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
