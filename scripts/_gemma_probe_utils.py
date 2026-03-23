#!/usr/bin/env python

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_SRC = REPO_ROOT / "third_party" / "openpi" / "src"
for p in (str(OPENPI_SRC),):
    if p not in sys.path:
        sys.path.insert(0, p)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
import openpi.policies.policy_config as policy_config  # noqa: E402
import openpi.training.config as training_config  # noqa: E402


@dataclasses.dataclass
class ProbeContext:
    config_name: str
    checkpoint_dir: Path
    repo_id: str
    device: str
    model: torch.nn.Module
    processor: Any


def _checkpoint_valid(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "params" / "_METADATA").exists()


def _format_checkpoint_hint(checkpoint_dir: Path) -> str:
    examples = [
        "~/.cache/openpi/openpi-assets/checkpoints/pi05_base",
        "<your_training_checkpoint_dir>/10000",
    ]
    lines = [
        f"Invalid checkpoint dir: {checkpoint_dir}",
        "Expected one of:",
        "- model.safetensors",
        "- params/_METADATA",
        "",
        "Likely issue: you passed tokenizer cache (e.g., big_vision) instead of a model checkpoint.",
        "Try paths like:",
    ]
    lines.extend([f"- {x}" for x in examples])
    return "\n".join(lines)


def load_probe_context(
    *,
    config_name: str,
    checkpoint_dir: str | Path,
    processor_id: str,
    repo_id_override: str | None = None,
    device: str | None = None,
) -> ProbeContext:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    if not _checkpoint_valid(checkpoint_dir):
        raise FileNotFoundError(_format_checkpoint_hint(checkpoint_dir))

    cfg = training_config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    repo_id = str(data_cfg.repo_id)
    if repo_id_override is not None and repo_id_override.strip():
        repo_id = repo_id_override.strip()
    if repo_id in ("", "None"):
        raise ValueError(
            f"Config '{config_name}' has empty repo_id. "
            "Please pass --repo-id (e.g. pipeline_ab/A)."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint_dir,
        norm_stats={},
        pytorch_device=device,
    )
    model = getattr(policy, "_model", None)
    if model is None:
        raise RuntimeError("Failed to access model from policy.")

    vlm = model.paligemma_with_expert.paligemma
    vlm = vlm.to(device)
    vlm.eval()
    # Attention probing requests attention maps; force eager attention backend to
    # avoid SDPA limitations and dtype mismatch in some runtime environments.
    for owner in (vlm, getattr(vlm, "model", None), getattr(getattr(vlm, "model", None), "language_model", None)):
        cfg = getattr(owner, "config", None)
        if cfg is None:
            continue
        setattr(cfg, "_attn_implementation", "eager")
        if hasattr(cfg, "attn_implementation"):
            setattr(cfg, "attn_implementation", "eager")

    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
    return ProbeContext(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        repo_id=repo_id,
        device=device,
        model=vlm,
        processor=processor,
    )


def load_dataset(repo_id: str, *, video_backend: str | None = None) -> LeRobotDataset:
    return LeRobotDataset(
        repo_id=repo_id,
        delta_timestamps=None,
        video_backend=video_backend,
    )


def resolve_head_camera_key(dataset: LeRobotDataset) -> str:
    keys = list(dataset.meta.camera_keys)
    preferred = [
        "observation.images.head_cam",
        "head_cam",
        "cam_high",
        "observation.images.top",
    ]
    for p in preferred:
        for key in keys:
            if key == p:
                return key
    for key in keys:
        low = key.lower()
        if "head" in low or "high" in low or "top" in low:
            return key
    if not keys:
        raise ValueError("No camera keys found in dataset.")
    return keys[0]


def episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    epi = getattr(dataset, "episode_data_index", None)
    if isinstance(epi, dict) and "from" in epi and "to" in epi:
        start = int(epi["from"][episode_index])
        end = int(epi["to"][episode_index])
    else:
        ep = dataset.meta.episodes[episode_index]
        if "dataset_from_index" in ep and "dataset_to_index" in ep:
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
        elif "length" in ep:
            # Fallback for metadata variants that only store per-episode length.
            start = int(sum(int(dataset.meta.episodes[i]["length"]) for i in range(episode_index)))
            end = start + int(ep["length"])
        else:
            raise KeyError(f"Cannot resolve episode bounds from metadata keys: {sorted(ep.keys())}")
    if end <= start:
        raise ValueError(f"Invalid episode span: episode={episode_index}, start={start}, end={end}")
    return start, end


def pil_from_frame(frame: Any) -> Image.Image:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.min() >= -1.01 and arr.max() <= 1.01:
            arr = ((arr + 1.0) * 127.5).clip(0, 255)
        elif arr.max() <= 1.01:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


def sample_last_frame(dataset: LeRobotDataset, episode_index: int, camera_key: str) -> tuple[Image.Image, int]:
    _, end = episode_bounds(dataset, episode_index)
    idx = end - 1
    item = dataset[idx]
    return pil_from_frame(item[camera_key]), idx


def _hf_row(dataset: LeRobotDataset, idx: int) -> dict[str, Any]:
    ensure = getattr(dataset, "_ensure_hf_dataset_loaded", None)
    if callable(ensure):
        ensure()  # noqa: SLF001
    return dataset.hf_dataset[idx]


def resolve_state_key(dataset: LeRobotDataset) -> str:
    ensure = getattr(dataset, "_ensure_hf_dataset_loaded", None)
    if callable(ensure):
        ensure()  # noqa: SLF001
    names = set(dataset.hf_dataset.column_names)
    for key in ("agent_pos", "observation.state", "state"):
        if key in names:
            return key
    raise ValueError(f"Cannot resolve state key. available={sorted(names)}")


def gripper_signal_from_state(state: Any) -> float:
    vec = np.asarray(state).reshape(-1)
    if vec.size >= 14:
        return float(np.mean(vec[[6, 13]]))
    if vec.size >= 7:
        return float(vec[6])
    return float(vec[-1])


def split_tail_by_gripper_close(dataset: LeRobotDataset, episode_index: int, state_key: str) -> tuple[list[int], list[int], int]:
    start, end = episode_bounds(dataset, episode_index)
    tail_start = start + (end - start) // 2
    tail_indices = list(range(tail_start, end))
    if len(tail_indices) < 2:
        return tail_indices[:1], tail_indices[-1:], tail_indices[-1]

    signal = []
    for idx in tail_indices:
        row = _hf_row(dataset, idx)
        signal.append(gripper_signal_from_state(row[state_key]))
    close_local = int(np.argmin(np.asarray(signal)))
    close_idx = tail_indices[close_local]

    pre = [i for i in tail_indices if i < close_idx]
    post = [i for i in tail_indices if i >= close_idx]
    if not pre and post:
        pre = post[:1]
    if not post and pre:
        post = pre[-1:]
    return pre, post, close_idx


def evenly_sample_indices(indices: list[int], n: int) -> list[int]:
    if n <= 0 or len(indices) <= n:
        return list(indices)
    sample_pos = np.linspace(0, len(indices) - 1, num=n)
    out = []
    for p in sample_pos:
        out.append(indices[int(round(float(p)))])
    return sorted(set(out))


def build_inputs(processor: Any, image: Image.Image, prompt: str, device: str) -> dict[str, torch.Tensor]:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def generate_answer(
    model: torch.nn.Module,
    processor: Any,
    *,
    image: Image.Image,
    prompt: str,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    inputs = build_inputs(processor, image, prompt, device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = float(temperature)

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = decoded.strip()
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    return text


def parse_xy(text: str, image_w: int, image_h: int) -> tuple[tuple[float, float] | None, bool]:
    # Prefer explicit x=..., y=... pattern.
    pattern = re.compile(
        r"x\s*[:=]\s*(-?\d+(?:\.\d+)?)\D+y\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
    else:
        # Fallback: take first two numeric values.
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(nums) < 2:
            return None, False
        x, y = float(nums[0]), float(nums[1])

    # Normalize [0, 1] outputs.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= image_w
        y *= image_h
    in_bounds = 0.0 <= x < image_w and 0.0 <= y < image_h
    return (x, y), in_bounds


def find_subsequence_positions(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    hits: list[int] = []
    for i in range(0, len(sequence) - len(pattern) + 1):
        if sequence[i : i + len(pattern)] == pattern:
            hits.extend(range(i, i + len(pattern)))
    return sorted(set(hits))


def choose_episode_indices(total_episodes: int, *, max_episodes: int, stride: int) -> list[int]:
    all_idx = list(range(total_episodes))
    sampled = all_idx[:: max(1, stride)]
    if max_episodes > 0:
        sampled = sampled[:max_episodes]
    return sampled


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized_entropy(values: np.ndarray, eps: float = 1e-12) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= eps:
        return 1.0
    p = vals / s
    p = np.clip(p, eps, 1.0)
    h = float(-(p * np.log(p)).sum())
    return h / math.log(len(p))
