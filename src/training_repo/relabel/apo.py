"""Minimal APO relabeling for offline samples."""

from __future__ import annotations

from typing import Any

from training_repo.common.schema import (
    LABEL_SOURCE_EXPERT,
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_PRE_INTERVENTION,
    SAMPLE_TYPE_CORRECT,
    SAMPLE_TYPE_INCORRECT,
    SAMPLE_TYPE_INTERACTION,
)


def relabel_episode_steps(
    episode: dict[str, Any],
    pre_intervention_k: int,
) -> list[dict[str, Any]]:
    if pre_intervention_k < 0:
        raise ValueError("pre_intervention_k must be >= 0")

    relabeled: list[dict[str, Any]] = []
    for step in episode["steps"]:
        sample_type = SAMPLE_TYPE_INTERACTION if step["intervention_flag"] else SAMPLE_TYPE_CORRECT
        label_source = LABEL_SOURCE_HUMAN if step["intervention_flag"] else LABEL_SOURCE_EXPERT
        relabeled.append(
            {
                "sample_id": f'{episode["episode_id"]}:{step["t"]}',
                "episode_id": episode["episode_id"],
                "t": step["t"],
                "obs_image_refs": step["obs_image_refs"],
                "obs_state": step["obs_state"],
                "action": step["action"],
                "intervention_flag": step["intervention_flag"],
                "terminal": step["terminal"],
                "sample_type": sample_type,
                "label_source": label_source,
            }
        )

    for idx, sample in enumerate(relabeled):
        current_is_intervention = sample["intervention_flag"]
        previous_is_intervention = relabeled[idx - 1]["intervention_flag"] if idx > 0 else False
        intervention_starts = current_is_intervention and not previous_is_intervention
        if not intervention_starts:
            continue

        start_idx = max(0, idx - pre_intervention_k)
        for relabel_idx in range(start_idx, idx):
            if relabeled[relabel_idx]["sample_type"] == SAMPLE_TYPE_INTERACTION:
                continue
            relabeled[relabel_idx]["sample_type"] = SAMPLE_TYPE_INCORRECT
            relabeled[relabel_idx]["label_source"] = LABEL_SOURCE_PRE_INTERVENTION

    return relabeled

