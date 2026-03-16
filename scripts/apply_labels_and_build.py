#!/usr/bin/env python
"""Apply A/B CSV labels to raw episodes, then build A/B dataset roots."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from training_repo.common.io import read_json, write_json, write_yaml
from training_repo.dataset_build.builder import build_dataset


LABEL_A = "A"
LABEL_B = "B"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply episode labels and build A/B datasets.")
    p.add_argument("--raw-data-root", type=Path, required=True, help="Input raw episode json root.")
    p.add_argument("--label-csv", type=Path, required=True, help="CSV from review GUI.")
    p.add_argument("--rewritten-raw-root", type=Path, required=True, help="Output root for rewritten jsons.")
    p.add_argument("--output-root", type=Path, required=True, help="Output root for built datasets.")
    p.add_argument(
        "--dataset-build-config-out",
        type=Path,
        default=None,
        help="Optional path prefix for generated dataset_build yamls.",
    )
    p.add_argument("--task-a-name", type=str, default="shirt_open_middle", help="Task name for label A.")
    p.add_argument("--task-b-name", type=str, default="shirt_flatten", help="Task name for label B.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-mode", type=str, default="episode")
    p.add_argument("--pre-intervention-k", type=int, default=3)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument(
        "--drop-non-ab",
        action="store_true",
        help="Drop episodes with labels not in {A, B} (recommended).",
    )
    return p.parse_args()


def _episode_number(value: str) -> int | None:
    match = re.search(r"(\d+)$", value.strip())
    if match is None:
        return None
    return int(match.group(1))


def _load_label_map(csv_path: Path) -> tuple[dict[str, str], dict[int, str]]:
    exact: dict[str, str] = {}
    by_num: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            episode_id = str(row.get("episode_id", "")).strip()
            label = str(row.get("label", "")).strip()
            if not episode_id:
                continue
            exact[episode_id] = label
            number = _episode_number(episode_id)
            if number is not None:
                by_num[number] = label
    return exact, by_num


def _resolve_label(episode_id: str, exact: dict[str, str], by_num: dict[int, str]) -> str:
    if episode_id in exact:
        return exact[episode_id]
    number = _episode_number(episode_id)
    if number is None:
        return ""
    return by_num.get(number, "")


def _mapped_task_name(label: str, task_a_name: str, task_b_name: str) -> str | None:
    if label == LABEL_A:
        return task_a_name
    if label == LABEL_B:
        return task_b_name
    return None


def main() -> None:
    args = parse_args()
    if not args.raw_data_root.exists():
        raise FileNotFoundError(f"raw_data_root not found: {args.raw_data_root}")
    if not args.label_csv.exists():
        raise FileNotFoundError(f"label_csv not found: {args.label_csv}")

    exact_labels, numeric_labels = _load_label_map(args.label_csv)
    src_files = sorted(args.raw_data_root.rglob("*.json"))
    if not src_files:
        raise ValueError(f"No json episodes found under: {args.raw_data_root}")

    kept = 0
    dropped = 0
    assigned_a = 0
    assigned_b = 0
    kept_non_ab = 0
    kept_a = 0
    kept_b = 0

    rewritten_all_root = args.rewritten_raw_root / "all_labeled"
    rewritten_a_root = args.rewritten_raw_root / "only_A"
    rewritten_b_root = args.rewritten_raw_root / "only_B"

    for src in src_files:
        episode = read_json(src)
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("episode_id", "")).strip()
        label = _resolve_label(episode_id, exact_labels, numeric_labels)
        mapped_task = _mapped_task_name(label, args.task_a_name, args.task_b_name)

        if mapped_task is None and args.drop_non_ab:
            dropped += 1
            continue
        if mapped_task is not None:
            episode["task_id"] = mapped_task
            if label == LABEL_A:
                assigned_a += 1
            elif label == LABEL_B:
                assigned_b += 1
        else:
            kept_non_ab += 1

        rel = src.relative_to(args.raw_data_root)
        write_json(rewritten_all_root / rel, episode)
        if label == LABEL_A:
            write_json(rewritten_a_root / rel, episode)
            kept_a += 1
        elif label == LABEL_B:
            write_json(rewritten_b_root / rel, episode)
            kept_b += 1
        kept += 1

    if kept == 0:
        raise ValueError("No episodes left after applying labels.")
    if assigned_a == 0 or assigned_b == 0:
        print(
            "warning: A/B label coverage is imbalanced or missing. "
            f"assigned_a={assigned_a}, assigned_b={assigned_b}"
        )

    if kept_a == 0 or kept_b == 0:
        raise ValueError(
            f"No labeled episodes for one side: kept_a={kept_a}, kept_b={kept_b}. "
            "Please complete A/B labels first."
        )

    cfg_prefix = args.dataset_build_config_out
    if cfg_prefix is None:
        cfg_prefix = args.output_root.parent / "dataset_build_from_labels"

    cfg_a = Path(str(cfg_prefix) + "_A.yaml")
    cfg_b = Path(str(cfg_prefix) + "_B.yaml")

    write_yaml(
        cfg_a,
        {
            "raw_data_root": str(rewritten_a_root.resolve()),
            "output_root": str((args.output_root / "A").resolve()),
            "schema_version": "v0.1.0",
            "val_ratio": float(args.val_ratio),
            "split_mode": str(args.split_mode).lower(),
            "pre_intervention_k": int(args.pre_intervention_k),
            "shard_size": int(args.shard_size),
            "random_seed": int(args.random_seed),
        },
    )
    write_yaml(
        cfg_b,
        {
            "raw_data_root": str(rewritten_b_root.resolve()),
            "output_root": str((args.output_root / "B").resolve()),
            "schema_version": "v0.1.0",
            "val_ratio": float(args.val_ratio),
            "split_mode": str(args.split_mode).lower(),
            "pre_intervention_k": int(args.pre_intervention_k),
            "shard_size": int(args.shard_size),
            "random_seed": int(args.random_seed),
        },
    )

    build_result_a = build_dataset(cfg_a)
    build_result_b = build_dataset(cfg_b)
    print("apply_labels_and_build complete")
    print(
        {
            "input_json_files": len(src_files),
            "kept_json_files": kept,
            "dropped_json_files": dropped,
            "assigned_a": assigned_a,
            "assigned_b": assigned_b,
            "kept_non_ab": kept_non_ab,
            "kept_a_json_files": kept_a,
            "kept_b_json_files": kept_b,
            "rewritten_all_root": str(rewritten_all_root.resolve()),
            "rewritten_a_root": str(rewritten_a_root.resolve()),
            "rewritten_b_root": str(rewritten_b_root.resolve()),
            "config_a_path": str(cfg_a),
            "config_b_path": str(cfg_b),
            "build_result_a": build_result_a,
            "build_result_b": build_result_b,
        }
    )


if __name__ == "__main__":
    main()
