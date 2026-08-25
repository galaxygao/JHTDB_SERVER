from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .config import load_config
from .grid import point_batches, time_chunks
from .jhtdb_client import JHTDBClient
from .pipeline import compute, validate_raw, write_report
from .verify import verify_results


def plan(config_path: str) -> dict[str, object]:
    cfg = load_config(config_path)
    spatial_batches = len(point_batches(cfg.point_count, cfg.max_points_per_query))
    temporal_batches = len(time_chunks(cfg.times, cfg.time_chunk_size))
    quantities = 2 + int(bool(cfg.gradient_audit))
    requests = spatial_batches * temporal_batches * quantities
    raw_bytes = cfg.point_count * len(cfg.times) * (3 + 9 * (quantities - 1)) * 4
    return {
        "dataset": cfg.dataset,
        "points": cfg.point_count,
        "point_batches": spatial_batches,
        "max_points_per_query": cfg.max_points_per_query,
        "snapshots": len(cfg.times),
        "time_chunks": temporal_batches,
        "queries_if_no_retry": requests,
        "strictly_serial": True,
        "raw_array_estimate_MiB": round(raw_bytes / 1024**2, 2),
        "core_shape": cfg.core_shape,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JHTDB testing-token acceleration regime pipeline")
    parser.add_argument(
        "command",
        choices=("plan", "smoke", "fetch", "validate", "verify", "compute", "classify", "report", "run"),
    )
    parser.add_argument("config", nargs="?", default="configs/task0.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "plan":
        print(json.dumps(plan(args.config), ensure_ascii=False, indent=2))
    elif args.command == "smoke":
        print(json.dumps(JHTDBClient(cfg).smoke(), ensure_ascii=False, indent=2))
    elif args.command == "fetch":
        print(JHTDBClient(cfg).fetch_all())
    elif args.command == "validate":
        print(json.dumps(validate_raw(cfg), ensure_ascii=False, indent=2))
    elif args.command == "verify":
        report, _, md_path = verify_results(cfg)
        print(f"{report['overall']}: {md_path}")
    elif args.command in ("compute", "classify"):
        print(compute(cfg))
    elif args.command == "report":
        print("\n".join(map(str, write_report(cfg))))
    elif args.command == "run":
        print(json.dumps(plan(args.config), ensure_ascii=False, indent=2))
        JHTDBClient(cfg).fetch_all()
        validate_raw(cfg)
        print(compute(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
