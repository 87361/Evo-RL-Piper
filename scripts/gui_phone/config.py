"""
Web Console Configuration
All security-sensitive and deployment-related settings are centralized here.
"""
import os
import secrets

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = int(os.environ.get("WEB_PORT", 3389))

# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────
# Change this password before first deployment!
# Can also be set via environment variable.
AUTH_PASSWORD = os.environ.get("WEB_PASSWORD", "wbcd2026")
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # 7 days

# ──────────────────────────────────────────────
# Whitelisted directories for dataset browsing
# ──────────────────────────────────────────────
DATASET_ROOTS = [
    # Add your actual dataset directories here
    "/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD",
]

# ──────────────────────────────────────────────
# Episode Review — dataset scanning root
# ──────────────────────────────────────────────
# Root directory for scanning LeRobot datasets (meta/info.json).
# Also used as the default scan root for merge sources.
EPISODE_SCAN_ROOTS = [
    "/data/vepfs/users/intern/lingyue.yang/datasets/WBCD/WBCD",
]

# ──────────────────────────────────────────────
# Training templates directory
# ──────────────────────────────────────────────
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# ──────────────────────────────────────────────
# Job management
# ──────────────────────────────────────────────
JOBS_DIR = os.path.join(os.path.dirname(__file__), ".jobs")
MAX_LOG_LINES = 500  # default tail lines for log endpoint

# ──────────────────────────────────────────────
# Training pipeline paths (shared with desktop GUI)
# ──────────────────────────────────────────────
# Project root: Evo-RL-Piper
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[2]
# OpenPI root (vendored JAX/PyTorch)
OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"

# Available OpenPI training configs
TRAIN_CONFIG_NAMES = [
    "pi05_aloha_wbcd_lora",
    "pi05_aloha_wbcd_4cam_lora",
    "pi0_aloha_pen_uncap",
    "pi05_aloha_pen_uncap",
]

# LeRobot policy types
# "lightweight" policies can be trained with `python src/lerobot/scripts/lerobot_train.py`
# "heavy" policies require `uv run --extra <name> python ...` for extra deps
LEROBOT_POLICY_TYPES = {
    "act":       {"label": "ACT",       "kind": "lightweight", "default_batch": 8, "default_steps": 100000},
    "diffusion": {"label": "Diffusion", "kind": "lightweight", "default_batch": 8, "default_steps": 400000},
    "smolvla":   {"label": "SmolVLA",   "kind": "heavy",       "default_batch": 32, "default_steps": 100000, "extra": "smolvla"},
    "xvla":      {"label": "XVLA",      "kind": "heavy",       "default_batch": 32, "default_steps": 100000, "extra": "xvla"},
}
