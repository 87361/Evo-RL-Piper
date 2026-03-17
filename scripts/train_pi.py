#!/usr/bin/env python
"""Launch PI-series training via third_party/openpi."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run PI training using third_party/openpi (JAX or PyTorch)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional launcher config yaml.",
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        type=str,
        default=None,
        help="OpenPI config name, e.g. pi0_aloha_sim / pi05_aloha / debug.",
    )
    parser.add_argument(
        "--backend",
        choices=["openpi_jax", "openpi_torch"],
        default="openpi_torch",
        help="Choose OpenPI training backend.",
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=Path("third_party/openpi"),
        help="Path to vendored OpenPI repository root.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def load_launcher_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Launcher config must be a mapping: {path}")
    return data


def main() -> None:
    args, passthrough = parse_args()
    launcher_cfg = load_launcher_config(args.config)

    backend = str(launcher_cfg.get("backend", args.backend))
    config_name = args.config_name or launcher_cfg.get("config_name")
    if not config_name:
        raise ValueError("Missing OpenPI config name. Provide positional config_name or set config_name in yaml.")
    openpi_root = Path(launcher_cfg.get("openpi_root", str(args.openpi_root))).resolve()
    script_rel = "scripts/train.py" if backend == "openpi_jax" else "scripts/train_pytorch.py"
    script_path = openpi_root / script_rel
    if not script_path.exists():
        raise FileNotFoundError(
            f"OpenPI training entry not found: {script_path}. "
            f"Please ensure third_party/openpi is initialized."
        )

    config_extra_args = launcher_cfg.get("extra_args", [])
    if not isinstance(config_extra_args, list):
        raise ValueError("extra_args in launcher config must be a list of strings.")
    command = [sys.executable, str(script_path), str(config_name), *[str(x) for x in config_extra_args], *passthrough]
    print("launching openpi training")
    print({"backend": backend, "entry": str(script_path), "argv": command[2:]})
    subprocess.run(command, check=True, cwd=str(openpi_root))


if __name__ == "__main__":
    main()
