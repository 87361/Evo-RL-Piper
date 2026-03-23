#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from routes import app_factory as _app_factory


def app_factory(
    video_root: Path,
    label_csv: Path,
    categories_json_override: Path | None = None,
    merge_sources: list[dict] | None = None,
    datasets_scan_root: Path | None = None,
) -> FastAPI:
    return _app_factory(
        video_root,
        label_csv,
        categories_json_override,
        merge_sources,
        datasets_scan_root,
    )
