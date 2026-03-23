"""Training task manager: SQLite state + tmux execution."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from wbcd_claw.config import AppConfig

_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    config_file TEXT NOT NULL,
    config_name TEXT NOT NULL,
    backend     TEXT NOT NULL,
    exp_name    TEXT NOT NULL,
    tmux_session TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    exit_code   INTEGER,
    created_at  TEXT NOT NULL,
    finished_at TEXT,
    extra_args  TEXT
);
"""


class TrainManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_path = config.db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(_DDL)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ---- config discovery ----

    def list_configs(self) -> list[dict]:
        results = []
        configs_dir = self.config.configs_dir
        if not configs_dir.is_absolute():
            configs_dir = (self.config.project_root / configs_dir).resolve()
        for p in sorted(configs_dir.glob("train_pi*_openpi.yaml")):
            cfg = self._parse_config(p)
            if cfg:
                results.append(cfg)
        return results

    def _parse_config(self, path: Path) -> dict | None:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return None
        config_name = data.get("config_name", "")
        backend = str(data.get("backend", "openpi_jax"))
        extra_args = data.get("extra_args", [])
        exp_name = ""
        for arg in (extra_args or []):
            a = str(arg)
            if a.startswith("--exp-name="):
                exp_name = a.split("=", 1)[1]
        return {
            "file": path.name,
            "path": str(path),
            "config_name": config_name,
            "backend": backend,
            "exp_name": exp_name,
            "extra_args": extra_args,
        }

    # ---- launch ----

    def launch(self, config_file: str, exp_name: str = "") -> dict:
        if not shutil.which("tmux"):
            return {"ok": False, "error": "tmux not found on server"}

        configs_dir = self.config.configs_dir
        if not configs_dir.is_absolute():
            configs_dir = (self.config.project_root / configs_dir).resolve()
        cfg_path = configs_dir / config_file
        if not cfg_path.exists():
            return {"ok": False, "error": f"config not found: {config_file}"}

        parsed = self._parse_config(cfg_path)
        if not parsed:
            return {"ok": False, "error": f"failed to parse config: {config_file}"}

        actual_exp_name = exp_name.strip() or parsed["exp_name"] or "claw_run"
        task_id = datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        tmux_session = f"claw_{task_id}"

        extra_args = list(parsed["extra_args"] or [])
        new_extra = []
        for a in extra_args:
            s = str(a)
            if s.startswith("--exp-name="):
                continue
            new_extra.append(s)
        new_extra.append(f"--exp-name={actual_exp_name}")

        train_script = str(self.config.project_root / "scripts" / "train_pi.py")
        cmd_parts = [
            sys.executable, train_script,
            "--config", str(cfg_path),
            parsed["config_name"],
            *new_extra,
        ]
        shell_cmd = " ".join(cmd_parts)
        env_prefix = f"cd {self.config.project_root} && PYTHONPATH=src"
        full_cmd = f"{env_prefix} {shell_cmd}"

        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", tmux_session,
            f"bash -c '{full_cmd}; echo EXIT_CODE=$?; exec bash'",
        ]

        proc = subprocess.run(tmux_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return {"ok": False, "error": f"tmux launch failed: {proc.stderr}"}

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, config_file, config_name, backend, exp_name, "
                "tmux_session, status, created_at, extra_args) VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, config_file, parsed["config_name"], parsed["backend"],
                 actual_exp_name, tmux_session, "running", now,
                 str(new_extra)),
            )
            conn.commit()

        return {"ok": True, "task_id": task_id, "tmux_session": tmux_session}

    # ---- status ----

    def list_tasks(self, limit: int = 30) -> list[dict]:
        self._refresh_running_tasks()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_task(self, task_id: str) -> dict | None:
        self._refresh_running_tasks()
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def _refresh_running_tasks(self) -> None:
        with self._conn() as conn:
            running = conn.execute(
                "SELECT task_id, tmux_session FROM tasks WHERE status='running'"
            ).fetchall()

        for row in running:
            tid, session = row["task_id"], row["tmux_session"]
            if not self._tmux_session_alive(session):
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                exit_code = self._extract_exit_code(session)
                status = "completed" if exit_code == 0 else "failed"
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE tasks SET status=?, exit_code=?, finished_at=? WHERE task_id=?",
                        (status, exit_code, now, tid),
                    )
                    conn.commit()

    def _tmux_session_alive(self, session: str) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True,
        )
        return result.returncode == 0

    def _extract_exit_code(self, session: str) -> int | None:
        lines = self._capture_pane(session, 500)
        for line in reversed(lines):
            if line.strip().startswith("EXIT_CODE="):
                code_str = line.strip().split("=", 1)[1]
                if code_str.isdigit():
                    return int(code_str)
        return None

    # ---- logs ----

    def get_logs(self, task_id: str, lines: int = 80) -> dict:
        task = self.get_task(task_id)
        if not task:
            return {"ok": False, "error": "task not found"}
        session = task["tmux_session"]
        if not self._tmux_session_alive(session):
            log_lines = self._capture_pane(session, lines)
            if not log_lines:
                return {"ok": True, "lines": ["(session ended, no output captured)"], "status": task["status"]}
        else:
            log_lines = self._capture_pane(session, lines)
        return {"ok": True, "lines": log_lines, "status": task["status"]}

    def _capture_pane(self, session: str, count: int = 80) -> list[str]:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", f"-{count}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return []
        return result.stdout.splitlines()

    # ---- kill ----

    def kill_task(self, task_id: str) -> dict:
        task = self.get_task(task_id)
        if not task:
            return {"ok": False, "error": "task not found"}
        session = task["tmux_session"]
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status='killed', finished_at=? WHERE task_id=?",
                (now, task_id),
            )
            conn.commit()
        return {"ok": True}
