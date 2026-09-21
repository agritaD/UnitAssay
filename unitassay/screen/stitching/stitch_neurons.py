"""Cross-recording neuron identity matching.

Ports stitch_neurons.m. The 4 nested functions that capture parent scope
become methods on the NeuronStitcher class.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unitassay.screen.config import IsolateConfig
from unitassay.screen.io.ks_loader import load_ks_dir
from unitassay.screen.io.mat_reader import load_mat
from unitassay.screen.types import MatchDetail, StitchResult
from unitassay.screen.utils.psth import psth

logger = logging.getLogger(__name__)


class NeuronStitcher:
    """Cross-recording neuron identity matching.

    Identifies the same neuron across multiple recordings by correlating
    firing rate PSTHs and waveform shapes.  Channels are compared using
    Euclidean distance on probe geometry (microns) rather than electrode-
    count range.

    Parameters
    ----------
    cfg : pipeline configuration
    output_dirs : list of output directories, one per recording.
        Each must contain NPclu.mat, {clu}.SortData.mat,
        and {clu}.RasterData.mat.
    ks_dirs : list of Kilosort output directories, one per recording
        (matching output_dirs). Each must contain channel_map.npy and
        channel_positions.npy.
    stitch_output_dir : directory to save stitch results
    NPelec : optional probe geometry dict with keys 'elec_num', 'x_coord',
        'y_coord'.  If None, loaded from output_dirs or built from ks_dirs.
    """

    def __init__(
        self,
        cfg: IsolateConfig,
        output_dirs: list[str | Path],
        ks_dirs: list[str | Path],
        stitch_output_dir: str | Path,
        NPelec: dict[str, NDArray] | None = None,
    ):
        self.cfg = cfg
        self.output_dirs = [Path(d) for d in output_dirs]
        self.ks_dirs = [Path(d) for d in ks_dirs]
        self.stitch_output_dir = Path(stitch_output_dir)
        self.num_recs = len(self.output_dirs)

        assert len(self.ks_dirs) == self.num_recs, (
            f"ks_dirs must have same length as output_dirs ({self.num_recs})."
        )

        self.cluster_type = cfg.stitching.cluster_type
        self.fr_threshold = cfg.stitching.fr_corr_threshold
        self.wf_threshold = cfg.stitching.wf_corr_threshold
        self.min_recs = cfg.stitching.min_recordings
        self.range_um = cfg.stitching.channel_range_um
        self.top_k = cfg.stitching.top_k
        self.filter_responsive = cfg.stitching.filter_responsive
        self.responsive_threshold = cfg.stitching.responsive_threshold
        self.bn = tuple(cfg.raster.time_window)

        self.clu_info_per_rec: list[NDArray] = []
        self.chan_maps: list[NDArray] = []
        self.NPelec = NPelec

    def run(self) -> StitchResult:
        """Execute the stitching algorithm.

        Returns
        -------
        StitchResult with stitch_table [N x num_recs] and score matrices
        """
        self._load_NPelec()
        self._load_cluster_info()
        stitch_table, fr_scores, wf_scores = self._run_stitching()

        # Save results
        self.stitch_output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.stitch_output_dir / f"stitch_results_{self.cluster_type}.h5"
        try:
            import h5py
            with h5py.File(results_file, "w") as f:
                f.create_dataset("stitch_table", data=stitch_table)
                f.create_dataset("fr_scores", data=fr_scores)
                f.create_dataset("wf_scores", data=wf_scores)
                f.attrs["cluster_type"] = self.cluster_type
                for i, d in enumerate(self.output_dirs):
                    f.attrs[f"output_dir_{i}"] = str(d)
                for i, d in enumerate(self.ks_dirs):
                    f.attrs[f"ks_dir_{i}"] = str(d)
            logger.info("Stitch results saved to: %s", results_file)
        except ImportError:
            logger.warning("h5py not available; results not saved to disk.")

        return StitchResult(
            stitch_table=stitch_table,
            recs=[str(d) for d in self.output_dirs],
            cluster_type=self.cluster_type,
            fr_score_matrix=fr_scores,
            wf_score_matrix=wf_scores,
        )

    def _load_NPelec(self) -> None:
        """Load or build probe geometry (NPelec)."""
        if self.NPelec is not None:
            return

        # Try loading from output_dirs
        for odir in self.output_dirs:
            npelec_file = odir / "NPelec.mat"
            if npelec_file.exists():
                data = load_mat(npelec_file)
                np_elec_raw = data.get("NPelec", data)
                self.NPelec = _parse_NPelec(np_elec_raw)
                logger.info("Loaded NPelec from: %s", npelec_file)
                return

        # Build from first ks_dir
        self.NPelec = _build_NPelec_from_ks(self.ks_dirs[0])
        logger.info("Built NPelec from KS output: %s", self.ks_dirs[0])

        # Save to each output_dir that doesn't have it
        for odir in self.output_dirs:
            npelec_file = odir / "NPelec.mat"
            if not npelec_file.exists():
                try:
                    import h5py
                    with h5py.File(npelec_file, "w") as f:
                        grp = f.create_group("NPelec")
                        grp.create_dataset("elecNum", data=self.NPelec["elec_num"])
                        grp.create_dataset("x_coord", data=self.NPelec["x_coord"])
                        grp.create_dataset("y_coord", data=self.NPelec["y_coord"])
                    logger.info("Saved NPelec to: %s", npelec_file)
                except (ImportError, OSError):
                    pass

    def _load_cluster_info(self) -> None:
        """Load cluster info and channel map for each recording."""
        logger.info("Loading cluster info for %d recordings...", self.num_recs)
        logger.info("Stitching cluster type: %s", self.cluster_type)

        all_iso_chans: list[NDArray] = []

        for i_r in range(self.num_recs):
            odir = self.output_dirs[i_r]
            npclu_file = odir / "NPclu.mat"

            # Try HDF5 first, then .mat
            # File stores [2 x N] (MATLAB column-major); transpose to [N x 2]
            # so Python view matches MATLAB display.
            np_h5 = npclu_file.with_suffix(".h5")
            if np_h5.exists():
                import h5py
                with h5py.File(np_h5, "r") as f:
                    clu_info = np.array(f["Clu_info"]).T if "Clu_info" in f else np.empty((0, 2))
                    ks_clu_info = np.array(f["KSclu_info"]).T if "KSclu_info" in f else np.empty((0, 2))
                    iso_clu_info = None
                    for key in ("IsoClu_info", "iso_clu_info"):
                        if key in f:
                            iso_clu_info = np.array(f[key]).T
                            break
            elif npclu_file.exists():
                data = load_mat(npclu_file)
                clu_info = np.atleast_2d(data.get("Clu_info", np.empty((0, 2))))
                ks_clu_info = np.atleast_2d(data.get("KSclu_info", np.empty((0, 2))))
                iso_clu_info = np.atleast_2d(data["IsoClu_info"]) if "IsoClu_info" in data else None
            else:
                raise FileNotFoundError(
                    f"NPclu not found in {odir}\nRun pipeline_KS4 first."
                )

            # Select cluster type
            if self.cluster_type == "All":
                selected_info = clu_info
            elif self.cluster_type == "KS_Good":
                if ks_clu_info is None or len(ks_clu_info) == 0:
                    raise ValueError(
                        f"KSclu_info not found in {odir}. Check Kilosort output."
                    )
                selected_info = ks_clu_info
            elif self.cluster_type == "Isolated":
                if iso_clu_info is None or len(iso_clu_info) == 0:
                    raise ValueError(
                        f"IsoClu_info not found in {odir}. "
                        "Run auto_curate + extract_isolated_units first."
                    )
                selected_info = iso_clu_info
            else:
                raise ValueError(
                    f"cluster_type must be 'All', 'KS_Good', or 'Isolated', "
                    f"got '{self.cluster_type}'"
                )

            self.clu_info_per_rec.append(selected_info)

            # Load channel map from ks_dir
            sp = load_ks_dir(self.ks_dirs[i_r], exclude_noise=False)
            chan_map = sp.chan_map + 1  # 0->1 indexing (match MATLAB)
            self.chan_maps.append(chan_map)

            # Map channel indices to channel IDs
            # selected_info is [N x 2]: col 0 = cluster IDs, col 1 = channel indices (1-indexed)
            if len(selected_info) > 0:
                chan_indices = selected_info[:, 1].astype(int)
                valid_mask = (chan_indices >= 1) & (chan_indices <= len(chan_map))
                iso_chans = chan_map[chan_indices[valid_mask] - 1] if np.any(valid_mask) else np.array([], dtype=int)
            else:
                iso_chans = np.array([], dtype=int)
            all_iso_chans.append(iso_chans)

        self.all_iso_chans = np.concatenate(all_iso_chans) if all_iso_chans else np.array([])
        self.uni_chans = np.unique(self.all_iso_chans)
        logger.info("Found %d unique channels across recordings.", len(self.uni_chans))

    def _get_cluster_ids(self, channel_id: int) -> list[NDArray]:
        """Get cluster IDs on a given channel across all recordings."""
        result: list[NDArray] = []
        for i_r in range(self.num_recs):
            chan_map = self.chan_maps[i_r]
            chan_ind = np.where(chan_map == channel_id)[0]
            info = self.clu_info_per_rec[i_r]
            # info is [N x 2]: col 0 = cluster IDs, col 1 = channel indices (1-indexed)
            if len(info) == 0 or len(chan_ind) == 0:
                result.append(np.array([], dtype=np.int64))
                continue
            ch_match = np.isin(info[:, 1].astype(int), chan_ind + 1)
            result.append(info[ch_match, 0].astype(np.int64))
        return result

    def _get_within_range_channels(self, selected_channel: int) -> list[int]:
        """Get electrode numbers within channel_range_um microns of selected_channel.

        Uses Euclidean distance on probe geometry (NPelec).
        """
        elec_num = self.NPelec["elec_num"]
        x_coord = self.NPelec["x_coord"]
        y_coord = self.NPelec["y_coord"]

        chan_idx = np.where(elec_num == selected_channel)[0]
        if len(chan_idx) == 0:
            return []

        ci = chan_idx[0]
        dx = x_coord - x_coord[ci]
        dy = y_coord - y_coord[ci]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        neighbor_idx = np.where(dists <= self.range_um)[0]
        return elec_num[neighbor_idx].astype(int).tolist()

    def _get_waveform(self, clu: int, rec_idx: int) -> NDArray:
        """Get waveform for a cluster from SortData in output_dir."""
        sort_file = self.output_dirs[rec_idx] / f"{clu}.SortData.mat"
        # Try .mat and .h5
        for ext in (".mat", ".h5"):
            f = sort_file.with_suffix(ext)
            if f.exists():
                if ext == ".mat":
                    data = load_mat(f)
                    sd = data.get("SortData")
                    if sd is not None:
                        wf = _extract_wf(sd)
                        if wf is not None:
                            return wf
                else:
                    import h5py
                    with h5py.File(f, "r") as hf:
                        if "frame_0000" in hf and "clu_wf" in hf["frame_0000"]:
                            return np.array(hf["frame_0000"]["clu_wf"])
        return np.full(61, np.nan)

    def _get_firing_rate(self, clu: int, rec_idx: int) -> NDArray:
        """Get firing rate (PSTH) for a cluster from RasterData in output_dir."""
        smoothing = self.cfg.raster.smoothing
        spike_offset = getattr(self.cfg.raster, "spike_time_offset", 0.0)
        raster_file = self.output_dirs[rec_idx] / f"{clu}.RasterData.mat"
        for ext in (".mat", ".h5"):
            f = raster_file.with_suffix(ext)
            if f.exists():
                if ext == ".mat":
                    data = load_mat(f)
                    rd = data.get("RasterData")
                    if rd is not None:
                        spike_clu, _rt = _extract_raster(rd)
                        if spike_clu:
                            rate, _ = psth(spike_clu, self.bn, smoothing=smoothing,
                                           spike_time_offset=spike_offset)
                            return rate
                else:
                    import h5py
                    with h5py.File(f, "r") as hf:
                        if "spike_clu" in hf:
                            grp = hf["spike_clu"]
                            spike_clu = [np.array(grp[k]) for k in sorted(grp.keys())]
                            if spike_clu:
                                rate, _ = psth(spike_clu, self.bn, smoothing=smoothing,
                                               spike_time_offset=spike_offset)
                                return rate

        return np.full(self.bn[1] - self.bn[0] + 1, np.nan)

    def _run_stitching(self) -> tuple[NDArray, NDArray, NDArray]:
        """Execute the main stitching prediction loop.

        Returns
        -------
        stitch_table, fr_score_matrix, wf_score_matrix
        """
        logger.info("Running stitching prediction...")
        prediction_list: list[NDArray] = []
        fr_score_list: list[NDArray] = []
        wf_score_list: list[NDArray] = []

        for current_chan in self.uni_chans:
            nearby_chans = self._get_within_range_channels(int(current_chan))

            # Gather data for all nearby channels per recording
            all_cluster_ids: list[list[int]] = [[] for _ in range(self.num_recs)]
            all_wfs: list[list[NDArray]] = [[] for _ in range(self.num_recs)]
            all_rates: list[list[NDArray]] = [[] for _ in range(self.num_recs)]

            for ch in nearby_chans:
                cluster_ids = self._get_cluster_ids(int(ch))

                for i_rec in range(self.num_recs):
                    for c in cluster_ids[i_rec]:
                        clu_int = int(c)
                        fr = self._get_firing_rate(clu_int, i_rec)

                        # Filter non-responsive units if enabled
                        if self.filter_responsive and np.nanmean(fr) <= self.responsive_threshold:
                            continue

                        wf = self._get_waveform(clu_int, i_rec)
                        all_cluster_ids[i_rec].append(clu_int)
                        all_wfs[i_rec].append(wf)
                        all_rates[i_rec].append(fr)

            # Compare each cluster on current channel against other recordings
            current_clu_ids = self._get_cluster_ids(int(current_chan))

            for i_rec in range(self.num_recs):
                for clu_id in current_clu_ids[i_rec]:
                    clu_int = int(clu_id)
                    fr = self._get_firing_rate(clu_int, i_rec)

                    # Skip non-responsive units if enabled
                    if self.filter_responsive and np.nanmean(fr) <= self.responsive_threshold:
                        continue

                    wf = self._get_waveform(clu_int, i_rec)

                    stitched = np.full(self.num_recs, np.nan)
                    stitched[i_rec] = clu_id
                    fr_scores_row = np.full(self.num_recs, np.nan)
                    wf_scores_row = np.full(self.num_recs, np.nan)
                    fr_scores_row[i_rec] = 1.0
                    wf_scores_row[i_rec] = 1.0

                    other_recs = [r for r in range(self.num_recs) if r != i_rec]

                    for other_rec in other_recs:
                        other_ids = all_cluster_ids[other_rec]
                        if len(other_ids) == 0:
                            continue

                        other_wfs = all_wfs[other_rec]
                        other_rates = all_rates[other_rec]

                        # Compute correlations
                        fr_corrs = np.array([
                            _pairwise_corr(fr, r) for r in other_rates
                        ])
                        wf_corrs = np.array([
                            _pairwise_corr(wf, w) for w in other_wfs
                        ])

                        fr_corrs_clean = np.nan_to_num(fr_corrs, nan=-np.inf)
                        sorted_indices = np.argsort(-fr_corrs_clean)

                        best_idx = sorted_indices[0]
                        if (fr_corrs_clean[best_idx] >= self.fr_threshold
                                and wf_corrs[best_idx] >= self.wf_threshold):
                            stitched[other_rec] = other_ids[best_idx]
                            fr_scores_row[other_rec] = float(fr_corrs[best_idx])
                            wf_scores_row[other_rec] = float(wf_corrs[best_idx])

                    prediction_list.append(stitched)
                    fr_score_list.append(fr_scores_row)
                    wf_score_list.append(wf_scores_row)

        if not prediction_list:
            logger.info("No stitched neurons found.")
            empty = np.empty((0, self.num_recs))
            return empty, empty.copy(), empty.copy()

        # Deduplicate
        stitch_array = np.vstack(prediction_list)
        fr_score_array = np.vstack(fr_score_list)
        wf_score_array = np.vstack(wf_score_list)

        dedup = stitch_array.copy()
        dedup[np.isnan(dedup)] = 0
        _, unique_idx = np.unique(dedup, axis=0, return_index=True)
        sorted_unique_idx = np.sort(unique_idx)
        unique_stitch = stitch_array[sorted_unique_idx]
        unique_fr = fr_score_array[sorted_unique_idx]
        unique_wf = wf_score_array[sorted_unique_idx]

        # Filter by minimum recording count
        valid_counts = np.sum(~np.isnan(unique_stitch), axis=1)
        keep_mask = valid_counts >= self.min_recs
        stitch_table = unique_stitch[keep_mask]
        final_fr = unique_fr[keep_mask]
        final_wf = unique_wf[keep_mask]

        logger.info(
            "Stitching complete: %d neurons found across %d recordings.",
            stitch_table.shape[0], self.num_recs,
        )
        return stitch_table, final_fr, final_wf


def _pairwise_corr(a: NDArray, b: NDArray) -> float:
    """Compute Pearson correlation handling NaN pairwise."""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    min_len = min(len(a), len(b))
    if min_len == 0:
        return np.nan
    a, b = a[:min_len], b[:min_len]
    valid = ~(np.isnan(a) | np.isnan(b))
    if np.sum(valid) < 2:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _extract_wf(sort_data) -> NDArray | None:
    """Extract waveform from MATLAB SortData struct."""
    if isinstance(sort_data, dict):
        return np.atleast_1d(sort_data.get("CluWf", None))
    if isinstance(sort_data, np.ndarray) and sort_data.size > 0:
        first = sort_data.flat[0]
        if hasattr(first, "CluWf"):
            return np.atleast_1d(first.CluWf)
        if isinstance(first, dict):
            return np.atleast_1d(first.get("CluWf"))
    if hasattr(sort_data, "CluWf"):
        return np.atleast_1d(sort_data.CluWf)
    return None


def _extract_raster(raster_data):
    """Extract spike_clu and RT from MATLAB RasterData struct."""
    if isinstance(raster_data, dict):
        sc = raster_data.get("SpikeClu", [])
        rt = np.atleast_1d(raster_data.get("RT", []))
    elif hasattr(raster_data, "SpikeClu"):
        sc = raster_data.SpikeClu
        rt = np.atleast_1d(getattr(raster_data, "RT", []))
    elif isinstance(raster_data, np.ndarray) and raster_data.size > 0:
        first = raster_data.flat[0]
        if hasattr(first, "SpikeClu"):
            sc = first.SpikeClu
            rt = np.atleast_1d(getattr(first, "RT", []))
        else:
            return [], np.array([])
    else:
        return [], np.array([])

    # Convert to list
    if isinstance(sc, np.ndarray) and sc.dtype == object:
        sc = [np.atleast_1d(s) if s is not None else np.array([]) for s in sc.flat]
    elif not isinstance(sc, list):
        sc = [sc] if sc is not None else []
    return sc, rt


def _parse_NPelec(raw) -> dict[str, NDArray]:
    """Parse NPelec from a loaded .mat struct into a standard dict."""
    if isinstance(raw, dict):
        # Try various key conventions
        elec_num = raw.get("elecNum", raw.get("elec_num", raw.get("ElecNum")))
        x_coord = raw.get("x_coord", raw.get("X_coord", raw.get("xcoords")))
        y_coord = raw.get("y_coord", raw.get("Y_coord", raw.get("ycoords")))
    elif hasattr(raw, "elecNum"):
        elec_num = raw.elecNum
        x_coord = raw.x_coord
        y_coord = raw.y_coord
    elif isinstance(raw, np.ndarray) and raw.dtype.names:
        elec_num = raw["elecNum"].flatten()
        x_coord = raw["x_coord"].flatten()
        y_coord = raw["y_coord"].flatten()
    else:
        # Try structured array
        first = raw.flat[0] if isinstance(raw, np.ndarray) and raw.size > 0 else raw
        elec_num = getattr(first, "elecNum", None)
        x_coord = getattr(first, "x_coord", None)
        y_coord = getattr(first, "y_coord", None)

    return {
        "elec_num": np.atleast_1d(elec_num).flatten().astype(int),
        "x_coord": np.atleast_1d(x_coord).flatten().astype(float),
        "y_coord": np.atleast_1d(y_coord).flatten().astype(float),
    }


def _build_NPelec_from_ks(ks_dir: Path) -> dict[str, NDArray]:
    """Build probe geometry from Kilosort output files."""
    ks_dir = Path(ks_dir)
    chan_map_file = ks_dir / "channel_map.npy"
    chan_pos_file = ks_dir / "channel_positions.npy"

    if chan_map_file.exists() and chan_pos_file.exists():
        chan_map = np.load(chan_map_file).flatten()
        chan_pos = np.load(chan_pos_file)
        return {
            "elec_num": (chan_map + 1).astype(int),  # 0->1 indexing
            "x_coord": chan_pos[:, 0].astype(float),
            "y_coord": chan_pos[:, 1].astype(float),
        }

    raise FileNotFoundError(
        f"channel_map.npy and/or channel_positions.npy not found in {ks_dir}"
    )
