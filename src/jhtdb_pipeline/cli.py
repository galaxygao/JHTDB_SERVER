from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .auth import has_token, token_source
from .catalog import Catalog
from .config import load_config
from .doctor import doctor
from .jhtdb import fetch_snapshot, smoke
from .planning import plan
from .processing import finalize_result, process_center, resource_plan, upgrade_result
from .validation import validate_snapshot


DEFAULT_CONFIG = "configs/pipeline.yaml"


def _config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def _frame(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time-index", type=int, required=True)
    _config(parser)


def _sigma(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sigma-grid", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SciServer-only JHTDB periodic center pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    auth_parser = commands.add_parser("auth", help="report JHTDB token status")
    auth_parser.add_argument("action", choices=("status",))
    _config(auth_parser)

    doctor_parser = commands.add_parser("doctor", help="check SciServer environment")
    doctor_parser.add_argument("--time-index", type=int)
    _config(doctor_parser)

    for name in ("plan", "smoke", "cache", "validate-input", "status"):
        command = commands.add_parser(name)
        if name != "status":
            _frame(command)
        else:
            _config(command)

    for name in ("process-center", "finalize-result", "single-frame", "upgrade-result"):
        command = commands.add_parser(name)
        _frame(command)
        _sigma(command)

    gui = commands.add_parser("gui", help="start the read-only server GUI")
    gui.add_argument("--port", type=int, default=8501)
    _config(gui)
    return parser


def _status(cfg) -> dict[str, object]:
    inputs: list[dict[str, object]] = []
    if cfg.catalog_path.exists():
        with Catalog(cfg.catalog_path) as catalog:
            for row in catalog.snapshots(cfg.dataset):
                item = dict(row)
                item["tiles"] = catalog.tile_progress(cfg.dataset, row["time_index"])
                inputs.append(item)
    results: list[dict[str, object]] = []
    if cfg.result_root.exists():
        for path in sorted(cfg.result_root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            manifest_path = path / "manifest.json"
            complete = (path / "COMPLETE").is_file()
            item: dict[str, object] = {"result_id": path.name, "complete": complete}
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                item.update(
                    {
                        "time_index": manifest.get("time_index"),
                        "sigma_grid": manifest.get("sigma_grid"),
                        "manifest_status": manifest.get("status"),
                    }
                )
            results.append(item)
    return {"inputs": inputs, "results": results}


def _selected_sigmas(cfg, sigma_grid: float | None) -> tuple[float, ...]:
    return (float(sigma_grid),) if sigma_grid is not None else cfg.sigma_grids


def _run_single_frame(cfg, time_index: int, sigma_grid: float | None) -> list[Path]:
    report = doctor(cfg, time_index)
    if report["status"] != "ok":
        raise RuntimeError(f"SciServer doctor failed: {json.dumps(report['checks'])}")
    fetch_snapshot(cfg, time_index)
    validate_snapshot(cfg, time_index)
    results = []
    for sigma in _selected_sigmas(cfg, sigma_grid):
        print(f"batch frame={time_index} sigma_grid={sigma:g}", flush=True)
        process_center(cfg, time_index, sigma)
        results.append(finalize_result(cfg, time_index, sigma))
    return results


def _print_paths(paths: list[Path]) -> None:
    if len(paths) == 1:
        print(paths[0])
    else:
        print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "auth":
            print(
                json.dumps(
                    {
                        "configured": has_token(cfg),
                        "source": token_source(cfg),
                    }
                )
            )
        elif args.command == "doctor":
            report = doctor(cfg, args.time_index)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "ok" else 2
        elif args.command == "plan":
            payload = plan(cfg, args.time_index)
            payload["resources"] = resource_plan(cfg)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.command == "smoke":
            print(json.dumps(smoke(cfg, args.time_index), ensure_ascii=False, indent=2))
        elif args.command == "cache":
            print(fetch_snapshot(cfg, args.time_index))
        elif args.command == "validate-input":
            print(
                json.dumps(
                    validate_snapshot(cfg, args.time_index),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "process-center":
            _print_paths(
                [
                    process_center(cfg, args.time_index, sigma)
                    for sigma in _selected_sigmas(cfg, args.sigma_grid)
                ]
            )
        elif args.command == "finalize-result":
            _print_paths(
                [
                    finalize_result(cfg, args.time_index, sigma)
                    for sigma in _selected_sigmas(cfg, args.sigma_grid)
                ]
            )
        elif args.command == "upgrade-result":
            _print_paths(
                [
                    upgrade_result(cfg, args.time_index, sigma)
                    for sigma in _selected_sigmas(cfg, args.sigma_grid)
                ]
            )
        elif args.command == "single-frame":
            _print_paths(_run_single_frame(cfg, args.time_index, args.sigma_grid))
        elif args.command == "status":
            print(json.dumps(_status(cfg), ensure_ascii=False, indent=2))
        elif args.command == "gui":
            environment = os.environ.copy()
            environment["JHTDB_PIPELINE_CONFIG"] = str(Path(args.config).resolve())
            dashboard = Path(__file__).with_name("dashboard.py")
            command = [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard),
                "--server.address",
                "0.0.0.0",
                "--server.port",
                str(args.port),
            ]
            return subprocess.run(command, env=environment, check=False).returncode
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
