"""CLI entry points; the resource policy is installed before numerical imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from .config import load_config

STRATEGIES = ("random", "greedy", "lcb", "thompson")


def build_parser() -> argparse.ArgumentParser:
    """Return the ``run``/``validate``/``reproduce``/``benchmark``/``prepare-library`` parser."""
    parser = argparse.ArgumentParser(description="Reproducible active learning with official TabM")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate", "reproduce", "benchmark", "prepare-library"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                             help="Override a strict YAML key; paths remain relative to config")
        if name in {"reproduce", "benchmark", "prepare-library"}:
            command.add_argument("--output", type=Path, required=True)
        if name == "prepare-library":
            command.add_argument("--chunk-size", type=int, default=20_000,
                                 help="CSV records per worker task while building the store")
        if name == "benchmark":
            command.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
            command.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["random", "lcb"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute one command and return the process exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0].startswith("-") and args[0] not in {"-h", "--help"}:
        args.insert(0, "run")
    options = build_parser().parse_args(args)
    try:
        config = load_config(options.config, options.set)
        if options.command == "validate":
            print(json.dumps(config.to_dict(), indent=2))
        elif options.command == "run":
            from .resources import configure_resources

            runtime = configure_resources(config.resources, config.seed)
            from .pipeline import run_experiment

            print(json.dumps(run_experiment(config, runtime), indent=2))
        elif options.command == "prepare-library":
            from .config import ResourceConfig
            from .resources import configure_resources

            resources = ResourceConfig(cpu_threads=config.resources.cpu_threads, gpu_ids=config.resources.gpu_ids)
            configure_resources(resources, config.seed)
            from .store import build_library_store

            data = config.data
            metadata = build_library_store(
                data.path, options.output, id_column=data.id_column, smiles_column=data.smiles_column,
                score_column=data.score_column if config.mode == "replay" else None,
                invalid_policy=data.invalid_policy, deduplicate=data.deduplicate,
                fingerprint_type=data.fingerprint_type, radius=data.fingerprint_radius,
                n_bits=data.fingerprint_bits, workers=config.resources.cpu_threads, chunk_size=options.chunk_size,
            )
            print(json.dumps({key: metadata[key] for key in ("n_compounds", "n_rejected", "n_input_records",
                                                              "fingerprint", "files")}, indent=2))
        else:
            from .experiments import benchmark, reproduce

            if options.command == "reproduce":
                report = reproduce(config, options.output)
            else:
                report = benchmark(config, options.output, options.seeds, options.strategies)
            print(json.dumps(report, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
