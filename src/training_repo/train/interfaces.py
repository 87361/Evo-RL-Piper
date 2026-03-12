"""Training backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainResult:
    artifact_dir: Path
    final_loss: float
    epochs: int
    num_samples: int


class TrainingBackend(ABC):
    @abstractmethod
    def train(self, config: dict[str, Any]) -> TrainResult:
        raise NotImplementedError

