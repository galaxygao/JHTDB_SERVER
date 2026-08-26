from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import zarr

from .auth import delete_token, get_token, has_token, set_token
from .catalog import Catalog
from .config import load_config
from .gradients import (
    audit_gradients,
    compute_gradients,
    filtered_space_plan,
    gradient_space_plan,
    validate_divergence,
)
from .jhtdb import download_snapshot, preflight_space, smoke
from .physics import compute_snapshot, physics_space_plan
from .planning import plan
from .validation import validate_snapshot


DEFAULT_CONFIG = "configs/pipeline.yaml"


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JHTDB full-periodic velocity pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth", help="manage the local JHTDB token")
    auth_parser.add_argument("action", choices=("set", "status", "delete"))
    auth_parser.add_argument(
        "--show-token",
        action="store_true",
        help="include the plaintext token in auth status output",
    )
    _add_config(auth_parser)

    for name in (
        "plan",
        "smoke",
        "download",
        "validate",
        "gradient",
        "validate-divergence",
        "compute",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--time-index", type=int, required=True)
        _add_config(command)

    audit_parser = subparsers.add_parser("audit-gradient")
    audit_parser.add_argument("--time-index", type=int, required=True)
    audit_parser.add_argument("--size", type=int, default=32)
    audit_parser.add_argument(
        "--origin-xyz",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="0-based core origin; defaults to a centered block",
    )
    _add_config(audit_parser)

    status_parser = subparsers.add_parser("status")
    _add_config(status_parser)
    gui_parser = subparsers.add_parser("gui")
    _add_config(gui_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "auth":
            if args.action == "set":
                set_token(cfg)
                print("JHTDB token saved in the local system credential store.")
            elif args.action == "status":
                configured = has_token(cfg)
                payload = {"configured": configured, "backend": cfg.auth_backend}
                if args.show_token:
                    payload["token"] = get_token(cfg) if configured else None
                print(json.dumps(payload))
            else:
                print(json.dumps({"deleted": delete_token(cfg)}))
        elif args.command == "plan":
            payload = plan(cfg, args.time_index)
            payload["disk"] = preflight_space(cfg)
            payload["gradient_resources"] = gradient_space_plan(cfg)
            payload["spectral_preprocessing_resources"] = filtered_space_plan(cfg)
            payload["physics_resources"] = physics_space_plan(cfg)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.command == "smoke":
            print(json.dumps(smoke(cfg, args.time_index), ensure_ascii=False, indent=2))
        elif args.command == "download":
            print(download_snapshot(cfg, args.time_index))
        elif args.command == "validate":
            print(json.dumps(validate_snapshot(cfg, args.time_index), ensure_ascii=False, indent=2))
        elif args.command == "gradient":
            print(compute_gradients(cfg, args.time_index))
        elif args.command == "audit-gradient":
            origin = tuple(args.origin_xyz) if args.origin_xyz is not None else None
            print(
                json.dumps(
                    audit_gradients(cfg, args.time_index, args.size, origin),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "validate-divergence":
            report = validate_divergence(cfg, args.time_index)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["passed"]:
                return 2
        elif args.command == "compute":
            print(compute_snapshot(cfg, args.time_index))
        elif args.command == "status":
            if not cfg.catalog_path.exists():
                print("[]")
            else:
                with Catalog(cfg.catalog_path) as catalog:
                    payload = []
                    filtered_root = (
                        zarr.open_group(str(cfg.filtered_store_path), mode="r")
                        if cfg.filtered_store_path.exists()
                        else None
                    )
                    for row in catalog.snapshots(cfg.dataset):
                        item = dict(row)
                        item["tiles"] = catalog.tile_progress(cfg.dataset, row["time_index"])
                        item["gradients"] = catalog.gradient_progress(cfg.dataset, row["time_index"])
                        if filtered_root is None:
                            item["filtered"] = {"status": "missing"}
                        else:
                            try:
                                filtered = filtered_root[f"t{int(row['time_index']):06d}"]
                            except KeyError:
                                item["filtered"] = {"status": "missing"}
                            else:
                                item["filtered"] = {
                                    "status": filtered.attrs.get("status", "incomplete"),
                                    "manifest_hash": filtered.attrs.get("manifest_hash"),
                                    "filter_method": filtered.attrs.get("filter_method"),
                                }
                        payload.append(item)
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.command == "gui":
            environment = os.environ.copy()
            environment["JHTDB_PIPELINE_CONFIG"] = str(Path(args.config).resolve())
            dashboard = Path(__file__).with_name("dashboard.py")
            return subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], env=environment, check=False).returncode
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
