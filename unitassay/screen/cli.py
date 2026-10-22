"""Click CLI for the screen layer: ``python -m unitassay.screen run ...``

Subcommands:
  - ``run``     : Execute the full pipeline
  - ``convert`` : Convert between .mat and HDF5 formats

Authors: Agrita Dubey, Qiao (Betty) Xu
Developed with Claude (Anthropic).
Part of unitassay. Licence: MIT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="unitassay")
def cli() -> None:
    """unitassay screen: per-frame isolation quality from a sorter's output."""
    pass


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Path to YAML config file")
@click.option("--ks-dir", required=True, type=click.Path(), help="Path to Kilosort output directory")
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Path for pipeline output")
@click.option("--stages", "-s", multiple=True, default=None, help="Stages to run (default: extract through iso_units)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def run(
    config: str,
    ks_dir: str,
    output_dir: str,
    stages: tuple[str, ...] | None,
    verbose: bool,
) -> None:
    """Run the isolation pipeline over a sorter's output."""
    _setup_logging(verbose)

    from unitassay.screen.config import IsolateConfig
    from unitassay.screen.pipeline import Pipeline

    cfg = IsolateConfig.from_yaml(config)
    pipeline = Pipeline(cfg)

    stage_list = list(stages) if stages else None
    pipeline.run(
        ks_dir=ks_dir,
        output_dir=output_dir,
        stages=stage_list,
    )


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--direction", type=click.Choice(["mat2hdf5", "hdf52mat"]), default="mat2hdf5", help="Conversion direction")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def convert(
    input_path: str,
    output_path: str,
    direction: str,
    verbose: bool,
) -> None:
    """Convert between .mat and HDF5 formats."""
    _setup_logging(verbose)

    from unitassay.screen.io.converter import convert_file
    convert_file(Path(input_path), Path(output_path), direction)
    click.echo(f"Converted: {input_path} -> {output_path}")
