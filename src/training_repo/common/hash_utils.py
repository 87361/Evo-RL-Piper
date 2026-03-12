"""Hashing helpers for reproducible build metadata."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

