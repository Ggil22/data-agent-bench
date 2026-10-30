"""Command line interface: `dab <command>`."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_agent_bench.bench import datasets
from data_agent_bench.bench.report import write_report
from data_agent_bench.bench.runner import latest_records, load_records, run_benchmark
from data_agent_bench.bench.tasks import TASKS, select_tasks
from data_agent_bench.llm import MissingAPIKeyError, OpenAICompatClient, load_model_configs

DEFAULT_CONFIG = Path("configs/models.yaml")
DEFAULT_DATA = Path("data")
DEFAULT_RESULTS = Path("results")
DEFAULT_CACHE = Path(".cache/llm")


def cmd_data(args: argparse.Namespace) -> None:
    out = datasets.write(args.data_dir, seed=args.seed)
    print(f"Wrote datasets (seed={args.seed}) to {out}")


def cmd_tasks(args: argparse.Namespace) -> None:
    for t in TASKS:
        print(f"{t.id:<9} {t.category:<13} {t.difficulty:<7} {t.question}")
    print(f"\n{len(TASKS)} tasks")


def cmd_models(args: argparse.Namespace) -> None:
    for name, cfg in load_model_configs(args.config).items():
        ready = cfg.api_key_env is None or bool(os.environ.get(cfg.api_key_env))
        status = "ready" if ready else f"missing {cfg.api_key_env}"
        print(f"{name:<22} {cfg.provider:<11} {cfg.model:<45} {status}")


def cmd_run(args: argparse.Namespace) -> None:
    configs = load_model_configs(args.config)
    unknown = [m for m in args.models if m not in configs]
    if unknown:
        sys.exit(f"Unknown models: {', '.join(unknown)}. See `dab models`.")
    if not (args.data_dir / "sales.csv").exists():
        datasets.write(args.data_dir)

    cache_dir = None if args.no_cache else DEFAULT_CACHE

    def make_client(model: str, repeat: int) -> OpenAICompatClient:
        return OpenAICompatClient(
            configs[model], cache_dir=cache_dir, cache_salt=f"repeat={repeat}"
        )

    try:
        for m in args.models:  # fail fast on missing keys, before any work is done
            make_client(m, 0)
    except MissingAPIKeyError as exc:
        sys.exit(str(exc))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = DEFAULT_RESULTS / run_id
    results = run_benchmark(
        args.models,
        select_tasks(args.tasks),
        make_client,
        data_dir=args.data_dir,
        out_dir=out_dir,
        repeats=args.repeats,
        max_steps=args.max_steps,
    )
    report = write_report(latest_records(load_records(results)), out_dir, f"Run {run_id}")
    print(f"\nResults: {results}\nReport:  {report}")


def cmd_report(args: argparse.Namespace) -> None:
    records = latest_records(load_records(args.run_dir / "results.jsonl"))
    if not records:
        sys.exit(f"No results found in {args.run_dir}")
    print(f"Report: {write_report(records, args.run_dir, f'Run {args.run_dir.name}')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dab", description="Data analyst agent benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="Generate the synthetic datasets.")
    p.add_argument("--seed", type=int, default=datasets.DEFAULT_SEED)
    p.set_defaults(func=cmd_data)

    sub.add_parser("tasks", help="List benchmark tasks.").set_defaults(func=cmd_tasks)
    sub.add_parser("models", help="List configured models.").set_defaults(func=cmd_models)

    p = sub.add_parser("run", help="Run the benchmark.")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--tasks", default="all", help="'all', 'category:<name>' or comma-separated ids")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--run-id", help="Reuse an id to resume an interrupted run.")
    p.add_argument("--no-cache", action="store_true", help="Disable the LLM response cache.")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="Rebuild the report of a run.")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
