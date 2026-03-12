"""Typed constants for minimal training data schema."""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_TYPE_CORRECT = "correct"
SAMPLE_TYPE_INTERACTION = "interaction"
SAMPLE_TYPE_INCORRECT = "incorrect"

LABEL_SOURCE_EXPERT = "expert"
LABEL_SOURCE_HUMAN = "human_intervention"
LABEL_SOURCE_PRE_INTERVENTION = "pre_intervention_relabel"

ALL_SAMPLE_TYPES = {
    SAMPLE_TYPE_CORRECT,
    SAMPLE_TYPE_INTERACTION,
    SAMPLE_TYPE_INCORRECT,
}

ALL_LABEL_SOURCES = {
    LABEL_SOURCE_EXPERT,
    LABEL_SOURCE_HUMAN,
    LABEL_SOURCE_PRE_INTERVENTION,
}


@dataclass(frozen=True)
class DatasetSchemaVersion:
    version: str = "v0.1.0"

