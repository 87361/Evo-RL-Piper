"""Application configuration for WBCDClaw MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetEntry:
    name: str
    video_root: Path
    label_csv: Path


@dataclass
class AppConfig:
    datasets: list[DatasetEntry] = field(default_factory=list)
    configs_dir: Path = field(default_factory=lambda: Path("configs"))
    db_path: Path = field(default_factory=lambda: Path("wbcd_claw.db"))
    project_root: Path = field(default_factory=lambda: Path(".").resolve())
    password: str = ""
    host: str = "0.0.0.0"
    port: int = 18090
    cookie_name: str = "wbcd_claw_token"
    cookie_max_age: int = 86400 * 7  # 7 days

    # backward compat: single-dataset mode
    @property
    def video_root(self) -> Path:
        return self.datasets[0].video_root if self.datasets else Path(".")

    @property
    def label_csv(self) -> Path:
        return self.datasets[0].label_csv if self.datasets else Path("labels.csv")
