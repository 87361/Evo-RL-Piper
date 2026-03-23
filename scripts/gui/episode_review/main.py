#!/usr/bin/env python
"""Headless-friendly web GUI for episode task review (multi-category, grid view)."""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from app import app_factory

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode task review GUI")
    p.add_argument("--video-root", type=Path, required=True, help="Root dir containing episode_*.mp4")
    p.add_argument("--label-csv", type=Path, required=True, help="CSV output path for labels")
    p.add_argument(
        "--categories-json", type=Path, default=None,
        help="Shared categories JSON path (defaults to <label-csv-stem>_categories.json next to label-csv). "
             "Use the same path across multiple datasets to share categories.",
    )
    p.add_argument(
        "--merge-sources", nargs="*", default=[],
        help="Additional datasets for cross-dataset merge, format: name::dataset_root::label_csv. "
             "Can specify multiple.",
    )
    p.add_argument(
        "--datasets-scan-root", type=Path, default=None,
        help="Root directory to scan for discoverable LeRobot datasets. "
             "Defaults to two levels up from video-root (e.g. the WBCD parent dir).",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=18080)
    return p.parse_args()


def _parse_merge_source(raw: str) -> dict:
    parts = raw.split("::")
    if len(parts) < 2:
        raise ValueError(f"--merge-sources format: name::dataset_root[::label_csv], got: {raw}")
    name = parts[0].strip()
    ds_root = parts[1].strip()
    label_csv = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else str(Path(ds_root) / "task_labels.csv")
    return {"name": name, "dataset_root": ds_root, "label_csv": label_csv}


def main() -> None:
    args = parse_args()
    if not args.video_root.exists():
        raise FileNotFoundError(f"video root not found: {args.video_root}")

    cat_json = args.categories_json.resolve() if args.categories_json else None
    merge_sources = [_parse_merge_source(s) for s in args.merge_sources] if args.merge_sources else None

    # Default scan root: two levels up from video_root (video_root is typically dataset_root/videos)
    scan_root = args.datasets_scan_root
    if scan_root is None:
        scan_root = args.video_root.resolve().parent.parent.parent
    else:
        scan_root = scan_root.resolve()

    app = app_factory(
        args.video_root.resolve(),
        args.label_csv.resolve(),
        cat_json,
        merge_sources=merge_sources,
        datasets_scan_root=scan_root,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
