"""CLI entry point for WBCDClaw server."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from wbcd_claw.app import create_app
from wbcd_claw.config import AppConfig, DatasetEntry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WBCDClaw - Mobile-first data & training workstation")
    p.add_argument(
        "--dataset", action="append", default=[], metavar="NAME:VIDEO_ROOT:LABEL_CSV",
        help="Dataset in format 'name:video_root:label_csv'. Can be repeated. "
             "If label_csv is omitted, defaults to task_labels.csv next to video_root.",
    )
    # backward compat: single dataset mode
    p.add_argument("--video-root", type=Path, default=None, help="(single-dataset) video root dir")
    p.add_argument("--label-csv", type=Path, default=None, help="(single-dataset) label CSV file")

    p.add_argument("--configs-dir", type=Path, default=Path("configs"))
    p.add_argument("--db-path", type=Path, default=Path("wbcd_claw.db"))
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--password", type=str, default="", help="Access password (empty = no auth)")
    return p.parse_args()


def _parse_dataset_arg(raw: str) -> DatasetEntry:
    """Parse 'name:video_root' or 'name:video_root:label_csv'."""
    parts = raw.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"--dataset must be 'name:video_root[:label_csv]', got: {raw}")
    name = parts[0].strip()
    video_root = Path(parts[1].strip()).resolve()
    if len(parts) >= 3 and parts[2].strip():
        label_csv = Path(parts[2].strip()).resolve()
    else:
        label_csv = video_root.parent / "task_labels.csv"
    return DatasetEntry(name=name, video_root=video_root, label_csv=label_csv)


def main() -> None:
    args = parse_args()

    datasets: list[DatasetEntry] = []

    for raw in args.dataset:
        datasets.append(_parse_dataset_arg(raw))

    if not datasets and args.video_root:
        vr = args.video_root.resolve()
        lc = args.label_csv.resolve() if args.label_csv else vr.parent / "task_labels.csv"
        datasets.append(DatasetEntry(name="default", video_root=vr, label_csv=lc))

    if not datasets:
        raise SystemExit("Error: provide --dataset or --video-root")

    for ds in datasets:
        if not ds.video_root.exists():
            raise FileNotFoundError(f"video root not found: {ds.video_root}")

    config = AppConfig(
        datasets=datasets,
        configs_dir=args.configs_dir.resolve() if args.configs_dir.is_absolute() else args.configs_dir,
        db_path=args.db_path.resolve(),
        project_root=Path(".").resolve(),
        password=args.password,
        host=args.host,
        port=args.port,
    )

    print(f"WBCDClaw starting on http://{config.host}:{config.port}")
    for ds in datasets:
        print(f"  [{ds.name}] videos={ds.video_root}  labels={ds.label_csv}")
    print(f"  configs_dir: {config.configs_dir}")
    print(f"  auth: {'enabled' if config.password else 'disabled'}")

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
