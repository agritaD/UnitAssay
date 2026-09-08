#!/usr/bin/env python3
"""Run Kilosort4 spike sorting on a binary file.

This module is used in two ways:

1. As a Python function (imported by the Python pipeline):

    from unitassay.isolate.sorting.run_kilosort4 import run_kilosort4
    run_kilosort4(cfg, bin_file, ks_output_dir)

2. As a CLI script (called from MATLAB via run_kilosort4.m):

    python run_kilosort4.py --data <bin> --probe <file> --params <file> \
        --results <dir> --gpu <n>

Both paths call the same `run_kilosort4_impl()` function.

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Core implementation — used by both Python API and CLI
# ============================================================================

def run_kilosort4_impl(
    bin_file: str | Path,
    probe_file: str | Path,
    results_dir: str | Path,
    params_file: str | Path | None = None,
    gpu: int = 0,
) -> Path:
    """Run Kilosort4 on a single binary file.

    Parameters
    ----------
    bin_file : path to raw .ap.bin file
    probe_file : path to probe file (.mat, .prb, or .json)
    results_dir : output directory for KS4 results
    params_file : optional path to KS4 parameters JSON file (uses KS4 defaults if None)
    gpu : CUDA GPU device index

    Returns
    -------
    Path to results directory
    """
    try:
        import torch
        from kilosort.io import load_probe, BinaryFiltered
        from kilosort.gui.main import Context
        from kilosort.gui.sorter import KiloSortWorker
        from kilosort.preprocessing import get_whitening_matrix, get_highpass_filter
        from kilosort.utils import download_probes, PROBE_DIR
    except ImportError as e:
        raise ImportError(
            "kilosort package not installed. Install with: "
            "pip install 'kilosort[gui]==4.0.22'"
        ) from e

    data_path = Path(bin_file)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    assert data_path.exists(), f"Binary file not found: {data_path}"

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available, running on CPU (this will be very slow)")

    # Load probe — accepts .mat, .prb, or .json
    probe_path = Path(probe_file)
    if not probe_path.exists():
        # Fall back to kilosort's built-in probe directory
        download_probes()
        probe_path = Path(PROBE_DIR) / str(probe_file)
    assert probe_path.exists(), f"Probe file not found: {probe_path}"
    print(f"Probe: {probe_path}")
    probe_layout = load_probe(probe_path)

    print(f"Input:   {data_path}")
    print(f"Output:  {results_path}")

    if params_file is not None and str(params_file):
        # User-provided parameters via JSON file
        params_path = Path(params_file)
        assert params_path.exists(), f"Params file not found: {params_path}"
        print(f"Params:  {params_path}")

        with open(str(params_path)) as f:
            params = json.load(f)

        main_params, extra_params = params['main'], params['extra']
        all_params = {**main_params, **extra_params}
        all_params['clear_cache'] = False
        all_params['invert_sign'] = False
        all_params['save_preprocessed_copy'] = True
        all_params['do_CAR'] = True
        all_params['data_dtype'] = 'int16'

        n_channels = all_params["n_chan_bin"]
        sample_rate = all_params["fs"]
        cutoff = all_params['highpass_cutoff']
        chan_map = probe_layout["chanMap"]
        xc = probe_layout["xc"]
        yc = probe_layout["yc"]
        nskip = all_params["nskip"]
        tmin = all_params['tmin']
        tmax = all_params['tmax']
        artifact = all_params['artifact_threshold']
        shift = all_params['shift']
        scale = all_params['scale']

        print(f"Channels: {n_channels}, Sample rate: {sample_rate}")
        print("Starting Kilosort4 with custom parameters...")

        # Build binary file objects
        binary_file = BinaryFiltered(
            filename=str(data_path),
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            device=device,
            dtype='int16',
            tmin=tmin,
            tmax=tmax,
            artifact_threshold=artifact,
            shift=shift,
            scale=scale,
            file_object=None)

        highpass_filter = get_highpass_filter(fs=sample_rate, cutoff=cutoff, device=device)

        # Compute whitening matrix
        with BinaryFiltered(
            filename=str(data_path),
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            hp_filter=highpass_filter,
            device=device,
            dtype='int16',
            tmin=tmin,
            tmax=tmax,
            artifact_threshold=artifact,
            shift=shift,
            scale=scale,
            file_object=None) as bf:
            whitening_matrix = get_whitening_matrix(f=bf, xc=xc, yc=yc, nskip=nskip)

        # Filtered binary with whitening
        filt_binary_file = BinaryFiltered(
            filename=str(data_path),
            n_chan_bin=n_channels,
            fs=sample_rate,
            chan_map=chan_map,
            hp_filter=highpass_filter,
            whiten_mat=whitening_matrix,
            device=device,
            dtype='int16',
            tmin=tmin,
            tmax=tmax,
            artifact_threshold=artifact,
            shift=shift,
            scale=scale,
            file_object=None)

        # Create context and run
        context = Context(
            context_path=results_path,
            probe=probe_layout,
            probe_name=probe_path.name,
            raw_probe=probe_layout.copy(),
            params=all_params,
            data_path=data_path,
            binary_file=binary_file,
            whitening_matrix=whitening_matrix,
            filt_binary_file=filt_binary_file,
        )

        worker = KiloSortWorker(
            context=context,
            results_directory=results_path,
            steps=['spikesort', 'plots'],
            device=device,
            file_object=None)

        worker.run()

    else:
        # Use KS4 defaults via the simple API
        print("Params:  KS4 defaults")
        print("Starting Kilosort4 with default parameters...")

        from kilosort import run_kilosort

        run_kilosort(
            settings={
                "n_chan_bin": 385,
                "data_dir": str(data_path.parent),
                "results_dir": str(results_path),
            },
            filename=str(data_path),
            results_dir=results_path,
            probe=probe_layout,
        )

    print("Kilosort4 completed successfully.")
    return results_path


# ============================================================================
# Python API — called by mind_snag.pipeline
# ============================================================================

def run_kilosort4(cfg, bin_file, ks_output_dir):
    """Execute Kilosort4 on a SpikeGLX recording.

    Parameters
    ----------
    cfg : MindSnagConfig
        Uses cfg.probe_file, cfg.ks_params_file, cfg.gpu
    bin_file : path to raw .ap.bin file
    ks_output_dir : directory for KS4 results

    Returns
    -------
    Path to KS4 results directory
    """
    probe_file = getattr(cfg, "probe_file", "")
    if not probe_file:
        raise ValueError(
            "cfg.probe_file must be set to a probe file (.mat, .prb, or .json)"
        )

    params_file = getattr(cfg, "ks_params_file", None) or None
    gpu = getattr(cfg, "gpu", 0)

    return run_kilosort4_impl(
        bin_file=bin_file,
        probe_file=probe_file,
        results_dir=ks_output_dir,
        params_file=params_file,
        gpu=gpu,
    )


# ============================================================================
# CLI — called from MATLAB run_kilosort4.m via system()
# ============================================================================

def main():
    """CLI entry point — parses args and calls run_kilosort4_impl."""
    # Support @response_file syntax
    sys_args = []
    for arg in sys.argv[1:]:
        if arg.startswith('@'):
            with open(arg[1:]) as f:
                sys_args.extend([line.strip() for line in f.readlines()])
        else:
            sys_args.append(arg)

    parser = argparse.ArgumentParser(description="Run Kilosort4 spike sorting")
    parser.add_argument("--data", "-d", required=True,
                        help="Path to raw binary file (.ap.bin)")
    parser.add_argument("--results", "-r", required=True,
                        help="Output directory for KS4 results")
    parser.add_argument("--probe", "-p", required=True,
                        help="Path to probe file (.mat, .prb, or .json)")
    parser.add_argument("--params", "-a", default=None,
                        help="Path to KS4 parameters JSON file (optional)")
    parser.add_argument("--gpu", "-g", type=int, default=0,
                        help="CUDA GPU device index")
    args = parser.parse_args(sys_args)

    run_kilosort4_impl(
        bin_file=args.data,
        probe_file=args.probe,
        results_dir=args.results,
        params_file=args.params,
        gpu=args.gpu,
    )


if __name__ == '__main__':
    main()
