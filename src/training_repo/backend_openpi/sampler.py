"""Bucket-proportional sampler for APO sample types."""

from __future__ import annotations

import random
from dataclasses import dataclass

from training_repo.common.schema import (
    SAMPLE_TYPE_CORRECT,
    SAMPLE_TYPE_INCORRECT,
    SAMPLE_TYPE_INTERACTION,
)


@dataclass(frozen=True)
class SamplerConfig:
    batch_size: int
    ratio_correct: int
    ratio_incorrect: int
    ratio_interaction: int
    random_seed: int


def _compute_bucket_counts(cfg: SamplerConfig) -> dict[str, int]:
    ratio_sum = cfg.ratio_correct + cfg.ratio_incorrect + cfg.ratio_interaction
    if ratio_sum <= 0:
        raise ValueError("At least one ratio must be > 0.")

    raw_counts = {
        SAMPLE_TYPE_CORRECT: cfg.batch_size * cfg.ratio_correct / ratio_sum,
        SAMPLE_TYPE_INCORRECT: cfg.batch_size * cfg.ratio_incorrect / ratio_sum,
        SAMPLE_TYPE_INTERACTION: cfg.batch_size * cfg.ratio_interaction / ratio_sum,
    }

    counts = {k: int(v) for k, v in raw_counts.items()}
    remainders = sorted(
        ((k, raw_counts[k] - counts[k]) for k in raw_counts),
        key=lambda item: item[1],
        reverse=True,
    )
    while sum(counts.values()) < cfg.batch_size:
        for bucket, _ in remainders:
            if sum(counts.values()) >= cfg.batch_size:
                break
            counts[bucket] += 1
    return counts


class ProportionalBucketSampler:
    def __init__(self, sample_types: list[str], cfg: SamplerConfig) -> None:
        self.sample_types = sample_types
        self.cfg = cfg
        self._rng = random.Random(cfg.random_seed)
        self._bucket_counts = _compute_bucket_counts(cfg)

        self._bucket_indices = {
            SAMPLE_TYPE_CORRECT: [],
            SAMPLE_TYPE_INCORRECT: [],
            SAMPLE_TYPE_INTERACTION: [],
        }
        for idx, sample_type in enumerate(sample_types):
            if sample_type not in self._bucket_indices:
                continue
            self._bucket_indices[sample_type].append(idx)

        for bucket in self._bucket_indices:
            if self._bucket_counts[bucket] > 0 and not self._bucket_indices[bucket]:
                raise ValueError(f"Bucket {bucket} requested by ratio but no samples in split.")

    def iter_batches(self, num_steps: int) -> list[list[int]]:
        batches: list[list[int]] = []
        cursors = {bucket: 0 for bucket in self._bucket_indices}
        shuffled = {bucket: indices.copy() for bucket, indices in self._bucket_indices.items()}

        for bucket, indices in shuffled.items():
            self._rng.shuffle(indices)

        for _ in range(num_steps):
            batch_indices: list[int] = []
            for bucket, count in self._bucket_counts.items():
                if count <= 0:
                    continue
                indices = shuffled[bucket]
                for _ in range(count):
                    if cursors[bucket] >= len(indices):
                        cursors[bucket] = 0
                        self._rng.shuffle(indices)
                    batch_indices.append(indices[cursors[bucket]])
                    cursors[bucket] += 1
            self._rng.shuffle(batch_indices)
            batches.append(batch_indices)
        return batches

