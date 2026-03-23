#!/usr/bin/env python
"""Compatibility entry for episode review GUI.

This thin wrapper keeps the original command path stable:
`python scripts/review_episode_tasks_gui.py ...`
"""

from __future__ import annotations

import sys
from pathlib import Path

_GUI_DIR = Path(__file__).resolve().parent / "gui" / "episode_review"
if str(_GUI_DIR) not in sys.path:
    sys.path.insert(0, str(_GUI_DIR))

from app import app_factory  # noqa: E402
from main import parse_args, main  # noqa: E402


if __name__ == "__main__":
    main()
